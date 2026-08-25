"""Rendering-guided flow-matching inference runner with 3DGS-style densification.

Houses :class:`_GuidedRun` and its densifiable Gaussian-state helpers -- the
stateful engine behind :func:`surflo.inference.engine.guided_inference`. All of
the guidance and densification machinery lives in the runner, so the public
entry point and the Euler loop in :mod:`surflo.inference.engine` stay a short
ODE driver.

Per-camera ray geometry (:class:`CameraGeometryCache`), the per-step lambda
schedules and the curvature targets are precomputed at startup; the inner
render loop only indexes into them. Keep it that way -- these are hot paths run
once per camera per inner step. ODE intermediates are stored only when
``save_intermediates`` is set.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import numpy as np
import torch
from torch.optim import Adam

from surflo.structures.cameras import get_cameras_from_intrinsics_and_extrinsics

from .gaussians import (
    Gaussians,
    apply_camera_pose_to_gaussians,
    get_gaussian_parameters_from_point_cloud,
    get_isotropic_gaussian_parameters_from_point_cloud,
    quaternion_rotate_vectors,
)
from surflo.rendering.surflo import render_surflo
from surflo.structures.cameras import get_cameras_spatial_extent
from .losses import (
    CameraGeometryCache,
    compute_depth_order_loss,
    convert_features_to_normals,
    depth_loss,
    depth_to_normal_with_mask_cached,
    dn_loss_cached,
    get_biphase_time_grid,
    inverse_sigmoid,
    normal_alignment_loss_cached,
    normal_to_curvature,
    rgb_loss,
)
from .camera_refine import (
    ViewSampler,
    build_confidence_masks,
    build_cull_masks,
    build_vggt_depths,
    fill_gt_with_random_bg,
    precompute_lambda_schedule,
    upsample_supervision,
)
from .engine import PhaseTimer, _OdeStep, materialize_deferred_losses

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3DGS-style densify state (unchanged behaviour, with a few micro-fixes)
# ---------------------------------------------------------------------------
def _get_expon_lr_func(
    lr_init: float, lr_final: float,
    lr_delay_steps: int = 0, lr_delay_mult: float = 1.0,
    max_steps: int = 1_000_000,
):
    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * np.sin(
                0.5 * np.pi * float(np.clip(step / lr_delay_steps, 0.0, 1.0))
            )
        else:
            delay_rate = 1.0
        t = float(np.clip(step / max_steps, 0.0, 1.0))
        log_lerp = np.exp(np.log(lr_init) * (1.0 - t) + np.log(lr_final) * t)
        return float(delay_rate * log_lerp)
    return helper


def _build_optimizer_groups(
    *,
    aux_xyz_offset: torch.Tensor,
    aux_log_scales: torch.Tensor,
    aux_quats: torch.Tensor,
    aux_colors: torch.Tensor,
    aux_logit_opacities: torch.Tensor,
    aux_normals: Optional[torch.Tensor],
    aux_colors_sh: Optional[torch.Tensor],
    aux_delta_xyz: torch.Tensor,
    aux_cam_quats: Optional[torch.Tensor],
    aux_cam_trans: Optional[torch.Tensor],
    aux_exposure_coeffs: Optional[torch.Tensor],
    spatial_lr_scale: float,
    aux_lr_multiplier: float,
    cam_rot_lr: float,
    cam_trans_lr: float,
    feature_dc_lr: float = 0.0025,
    opacity_lr: float = 0.05,
    scaling_lr: float = 0.005,
    rotation_lr: float = 0.001,
    gaussian_features_lr: float = 0.05 / 2.0,
    colors_sh_lr: float = 0.0025 / 20.0,
    exposure_coeffs_lr: float = 0.001,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    xyz_lr = 0.00016 * float(spatial_lr_scale)
    groups: List[Dict[str, Any]] = []
    group_idx: Dict[str, int] = {}

    def _add(name: str, params: torch.Tensor, lr: float,
             betas: Optional[Tuple[float, float]] = None) -> None:
        g: Dict[str, Any] = {"params": [params], "lr": lr, "name": name}
        if betas is not None:
            g["betas"] = betas
        group_idx[name] = len(groups)
        groups.append(g)

    _add("aux_xyz_offset", aux_xyz_offset, xyz_lr * aux_lr_multiplier)
    _add("aux_log_scales", aux_log_scales, scaling_lr * aux_lr_multiplier)
    _add("aux_quats", aux_quats, rotation_lr * aux_lr_multiplier)
    _add("aux_colors", aux_colors, feature_dc_lr * aux_lr_multiplier)
    _add("aux_logit_opacities", aux_logit_opacities, opacity_lr * aux_lr_multiplier)
    if aux_normals is not None:
        _add("aux_normals", aux_normals, gaussian_features_lr * aux_lr_multiplier)
    if aux_colors_sh is not None:
        _add("aux_colors_sh", aux_colors_sh, colors_sh_lr * aux_lr_multiplier)

    _add("aux_delta_xyz", aux_delta_xyz, xyz_lr * aux_lr_multiplier)
    if aux_cam_quats is not None:
        _add("aux_cam_quats", aux_cam_quats, cam_rot_lr)
    if aux_cam_trans is not None:
        _add("aux_cam_trans", aux_cam_trans, cam_trans_lr)
    if aux_exposure_coeffs is not None:
        _add("aux_exposure_coeffs", aux_exposure_coeffs,
             exposure_coeffs_lr * aux_lr_multiplier, betas=(0.9, 0.99))
    return groups, group_idx


_PER_GAUSSIAN_GROUP_NAMES: Tuple[str, ...] = (
    "aux_xyz_offset", "aux_log_scales", "aux_quats", "aux_colors",
    "aux_logit_opacities", "aux_normals", "aux_colors_sh",
)


class GuidedGaussianAuxState:
    """Container for the flat per-Gaussian aux state used by
    :func:`guided_inference`.
    """

    def __init__(
        self, *,
        aux_xyz_offset: torch.nn.Parameter,
        aux_log_scales: torch.nn.Parameter,
        aux_quats: torch.nn.Parameter,
        aux_colors: torch.nn.Parameter,
        aux_logit_opacities: torch.nn.Parameter,
        aux_normals: Optional[torch.nn.Parameter],
        aux_colors_sh: Optional[torch.nn.Parameter],
        anchor_idx: torch.Tensor,
        aux_delta_xyz: torch.nn.Parameter,
        aux_cam_quats: Optional[torch.nn.Parameter],
        aux_cam_trans: Optional[torch.nn.Parameter],
        aux_exposure_coeffs: Optional[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        group_idx: Dict[str, int],
        num_anchors: int,
        num_sh: int,
        sh_degree: int,
        learn_normals: bool,
        scene_extent: float,
        device: torch.device,
        position_scheduler: Optional[Callable[[int], float]] = None,
        position_scheduled_groups: Tuple[str, ...] = (
            "aux_xyz_offset", "aux_delta_xyz",
        ),
        active_sh_degree: int = 0,
        sh_degree_warmup_interval: int = 0,
    ) -> None:
        self.aux_xyz_offset = aux_xyz_offset
        self.aux_log_scales = aux_log_scales
        self.aux_quats = aux_quats
        self.aux_colors = aux_colors
        self.aux_logit_opacities = aux_logit_opacities
        self.aux_normals = aux_normals
        self.aux_colors_sh = aux_colors_sh
        self.anchor_idx = anchor_idx
        self.aux_delta_xyz = aux_delta_xyz
        self.aux_cam_quats = aux_cam_quats
        self.aux_cam_trans = aux_cam_trans
        self.aux_exposure_coeffs = aux_exposure_coeffs
        self.optimizer = optimizer
        self.group_idx = group_idx
        self.num_anchors = int(num_anchors)
        self.num_sh = int(num_sh)
        self.sh_degree = int(sh_degree)
        self.learn_normals = bool(learn_normals)
        self.scene_extent = float(scene_extent)
        self.device = device

        self.position_scheduler = position_scheduler
        self.position_scheduled_groups = tuple(position_scheduled_groups)
        self.last_position_lr: Optional[float] = None

        self.max_sh_degree = int(self.sh_degree)
        self.active_sh_degree = int(min(int(active_sh_degree), self.max_sh_degree))
        self.sh_degree_warmup_interval = int(sh_degree_warmup_interval)

        N = int(aux_xyz_offset.shape[0])
        self.xyz_gradient_accum = torch.zeros(N, 1, device=device)
        self.xyz_gradient_accum_abs = torch.zeros(N, 1, device=device)
        self.denom = torch.zeros(N, 1, device=device)
        self.max_radii2D = torch.zeros(N, device=device)

        # Cached helper used by the velocity-update path; resized lazily.
        self._ones_for_anchor: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    @property
    def num_gaussians(self) -> int:
        return int(self.aux_xyz_offset.shape[0])

    def per_gaussian_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        out: List[Tuple[str, torch.nn.Parameter]] = []
        for name in _PER_GAUSSIAN_GROUP_NAMES:
            if name not in self.group_idx:
                continue
            param = getattr(self, name)
            if param is not None:
                out.append((name, param))
        return out

    def ones_for_anchor_count(self) -> torch.Tensor:
        N = self.num_gaussians
        if self._ones_for_anchor is None or self._ones_for_anchor.shape[0] != N:
            self._ones_for_anchor = torch.ones(N, 1, device=self.device)
        return self._ones_for_anchor

    # ------------------------------------------------------------------
    def update_learning_rate(self, iteration: int) -> Optional[float]:
        if self.position_scheduler is None:
            return None
        new_lr = float(self.position_scheduler(int(iteration)))
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") in self.position_scheduled_groups:
                param_group["lr"] = new_lr
        self.last_position_lr = new_lr
        return new_lr

    def oneup_sh_degree(self) -> int:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        return self.active_sh_degree

    def maybe_oneup_sh_degree(self, iteration: int) -> bool:
        if self.sh_degree_warmup_interval <= 0:
            return False
        if iteration % self.sh_degree_warmup_interval != 0:
            return False
        prev = self.active_sh_degree
        new = self.oneup_sh_degree()
        return new > prev

    # ------------------------------------------------------------------
    def make_gaussians(
        self,
        pts_0: torch.Tensor,
        anchor_normals: Optional[torch.Tensor],
        decouple_normals: bool,
    ) -> Tuple["Gaussians", Optional["Gaussians"], Optional[torch.Tensor]]:
        anchor_pts = pts_0 + self.aux_delta_xyz
        means_flat = anchor_pts[self.anchor_idx] + self.aux_xyz_offset

        scales_flat = self.aux_log_scales.exp()
        rotations_flat = torch.nn.functional.normalize(self.aux_quats, dim=-1)
        opacities_flat = self.aux_logit_opacities.sigmoid().reshape(-1)
        colors_flat = self.aux_colors
        colors_sh_flat = self.aux_colors_sh

        gs = Gaussians(
            means=means_flat, colors=colors_flat, scales=scales_flat,
            rotations=rotations_flat, opacities=opacities_flat,
            colors_sh=colors_sh_flat,
            active_sh_degree=int(self.active_sh_degree),
        )

        if decouple_normals:
            gs_nograd = Gaussians(
                means=means_flat.detach(), colors=colors_flat.detach(),
                scales=scales_flat.detach(), rotations=rotations_flat.detach(),
                opacities=opacities_flat.detach(),
                colors_sh=(colors_sh_flat.detach() if colors_sh_flat is not None else None),
                active_sh_degree=int(self.active_sh_degree),
            )
        else:
            gs_nograd = None

        if self.learn_normals and self.aux_normals is not None:
            nrm_per_gauss = convert_features_to_normals(features=self.aux_normals)
        elif anchor_normals is not None:
            nrm_per_gauss = anchor_normals[self.anchor_idx]
        else:
            nrm_per_gauss = None
        return gs, gs_nograd, nrm_per_gauss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def add_densification_stats(self, pkg: Dict[str, torch.Tensor]) -> None:
        vsp = pkg.get("viewspace_points")
        vis = pkg.get("visibility_filter")
        radii = pkg.get("radii")
        if vsp is None or vis is None or radii is None or vsp.grad is None:
            return
        if vsp.shape[0] != self.num_gaussians:
            return

        vis_b = vis.bool()
        self.max_radii2D[vis_b] = torch.maximum(
            self.max_radii2D[vis_b], radii[vis_b].float(),
        )
        self.xyz_gradient_accum[vis_b] += torch.norm(
            vsp.grad[vis_b, :2], dim=-1, keepdim=True,
        )
        if vsp.grad.shape[-1] >= 3:
            self.xyz_gradient_accum_abs[vis_b] += torch.norm(
                vsp.grad[vis_b, 2:], dim=-1, keepdim=True,
            )
        self.denom[vis_b] += 1

    # ------------------------------------------------------------------
    @torch.no_grad()
    def densify_and_prune(
        self, *,
        max_grad: float,
        min_opacity: float,
        max_screen_size: Optional[int],
        percent_dense: float,
        densify_split_n: int = 2,
        use_abs_grad: bool = True,
    ) -> Dict[str, int]:
        N_before = self.num_gaussians
        if N_before == 0:
            return {"n_clone": 0, "n_split": 0, "n_prune": 0,
                    "N_before": 0, "N_after": 0}

        denom_safe = self.denom.clamp_min(1.0)
        grads = torch.nan_to_num(
            self.xyz_gradient_accum / denom_safe,
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        grads_norm = grads.squeeze(-1)

        Q: Optional[torch.Tensor] = None
        grads_abs_norm: Optional[torch.Tensor] = None
        if use_abs_grad:
            grads_abs = torch.nan_to_num(
                self.xyz_gradient_accum_abs / denom_safe,
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            grads_abs_norm = grads_abs.squeeze(-1)
            # Avoid the .item() sync; use a pure-GPU ratio.
            ratio_t = (grads_norm >= max_grad).float().mean()
            # Shortcut: if ratio==0 nothing is selected via abs branch either.
            if grads_abs_norm.numel() > 0:
                # Compute Q on GPU; if ratio==0 set Q to +inf.
                Q = torch.quantile(
                    grads_abs_norm.reshape(-1),
                    (1.0 - ratio_t).clamp(0.0, 1.0),
                )
                Q = torch.where(
                    ratio_t > 0.0, Q,
                    torch.tensor(float("inf"), device=self.device),
                )
            else:
                Q = torch.tensor(float("inf"), device=self.device)

        scales = self.aux_log_scales.exp()
        max_scale = scales.max(dim=1).values

        clone_mask = grads_norm >= max_grad
        if Q is not None:
            clone_mask = clone_mask | (grads_abs_norm >= Q)
        clone_mask = clone_mask & (max_scale <= percent_dense * self.scene_extent)
        n_clone_t = clone_mask.sum()
        self._densify_clone(clone_mask)

        n_pre_split = self.num_gaussians

        def _pad_to(t: torch.Tensor, n: int) -> torch.Tensor:
            if n <= t.shape[0]:
                return t
            out = torch.zeros(n, device=t.device, dtype=t.dtype)
            out[: t.shape[0]] = t
            return out

        grads_padded = _pad_to(grads_norm, n_pre_split)
        scales = self.aux_log_scales.exp()
        max_scale = scales.max(dim=1).values
        split_mask = grads_padded >= max_grad
        if Q is not None:
            grads_abs_padded = _pad_to(grads_abs_norm, n_pre_split)
            split_mask = split_mask | (grads_abs_padded >= Q)
        split_mask = split_mask & (max_scale > percent_dense * self.scene_extent)
        n_split_t = self._densify_split(split_mask, N=int(densify_split_n))

        opacities = self.aux_logit_opacities.sigmoid().reshape(-1)
        prune_mask = opacities < min_opacity
        if max_screen_size is not None and max_screen_size > 0:
            big_screen = self.max_radii2D > max_screen_size
            scales = self.aux_log_scales.exp()
            big_world = scales.max(dim=1).values > 0.1 * self.scene_extent
            prune_mask = prune_mask | big_screen | big_world
        n_prune_t = prune_mask.sum()
        if int(n_prune_t.item()) > 0:
            self._prune_points(prune_mask)

        N = self.num_gaussians
        self.xyz_gradient_accum = torch.zeros(N, 1, device=self.device)
        self.xyz_gradient_accum_abs = torch.zeros(N, 1, device=self.device)
        self.denom = torch.zeros(N, 1, device=self.device)
        self.max_radii2D = torch.zeros(N, device=self.device)

        # Single sync at the end (was 3 separate .item() calls + empty_cache).
        n_clone = int(n_clone_t.item())
        n_prune = int(n_prune_t.item())
        return {
            "n_clone": n_clone,
            "n_split": int(n_split_t),
            "n_prune": n_prune,
            "N_before": int(N_before),
            "N_after": int(N),
        }

    def _densify_clone(self, mask: torch.Tensor) -> None:
        # mask is a bool tensor; sum on GPU and short-circuit on zero
        # without a forced sync (we just check a single element via .item()).
        if int(mask.sum().item()) == 0:
            return
        new_tensors: Dict[str, torch.Tensor] = {}
        for name, param in self.per_gaussian_params():
            new_tensors[name] = param.detach()[mask].clone()
        new_anchor = self.anchor_idx[mask].clone()
        self._cat_tensors_to_optimizer(new_tensors, new_anchor)

    def _densify_split(self, mask: torch.Tensor, N: int = 2) -> int:
        n_parents = int(mask.sum().item())
        if n_parents == 0 or N < 1:
            return 0
        device = self.device
        n_new = n_parents * N

        parent_offsets = self.aux_xyz_offset.detach()[mask]
        parent_scales = self.aux_log_scales.detach().exp()[mask]
        parent_quats = torch.nn.functional.normalize(
            self.aux_quats.detach()[mask], dim=-1,
        )
        parent_anchor = self.anchor_idx[mask]

        z = torch.randn(n_parents, N, 3, device=device, dtype=parent_offsets.dtype)
        z_scaled = z * parent_scales.unsqueeze(1)
        q_expand = parent_quats.unsqueeze(1).expand(n_parents, N, 4).reshape(-1, 4)
        z_rot = quaternion_rotate_vectors(
            q_expand, z_scaled.reshape(-1, 3),
        ).reshape(n_parents, N, 3)
        new_xyz_offset = (parent_offsets.unsqueeze(1) + z_rot).reshape(-1, 3)

        new_log_scales = torch.log(
            parent_scales.unsqueeze(1).expand(n_parents, N, 3).reshape(-1, 3)
            / (0.8 * float(N))
        )

        new_tensors: Dict[str, torch.Tensor] = {
            "aux_xyz_offset": new_xyz_offset,
            "aux_log_scales": new_log_scales,
        }
        for name, param in self.per_gaussian_params():
            if name in ("aux_xyz_offset", "aux_log_scales"):
                continue
            t = param.detach()[mask]
            shape_extra = t.shape[1:]
            t_rep = (
                t.unsqueeze(1)
                .expand(n_parents, N, *shape_extra)
                .reshape(n_parents * N, *shape_extra)
                .clone()
            )
            new_tensors[name] = t_rep
        new_anchor = parent_anchor.unsqueeze(1).expand(n_parents, N).reshape(-1).clone()

        self._cat_tensors_to_optimizer(new_tensors, new_anchor)
        N_after_cat = self.num_gaussians
        prune_old = torch.zeros(N_after_cat, dtype=torch.bool, device=device)
        prune_old[: mask.shape[0]] = mask
        self._prune_points(prune_old)
        return n_new

    @torch.no_grad()
    def reset_opacity(self, opacity_reset_value: float = 0.01) -> None:
        opac = self.aux_logit_opacities.sigmoid()
        capped = torch.min(
            opac, torch.full_like(opac, float(opacity_reset_value)),
        )
        new_logits = inverse_sigmoid(capped)
        self._replace_tensor_to_optimizer(new_logits, "aux_logit_opacities")

    def _prune_optimizer(self, valid_mask: torch.Tensor) -> None:
        for name in _PER_GAUSSIAN_GROUP_NAMES:
            if name not in self.group_idx:
                continue
            g_idx = self.group_idx[name]
            group = self.optimizer.param_groups[g_idx]
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)

            new_data = old_param.data[valid_mask]
            new_param = torch.nn.Parameter(new_data.requires_grad_(True))
            if stored_state is not None:
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                if "exp_avg_sq" in stored_state:
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]
                del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            setattr(self, name, new_param)

        self.anchor_idx = self.anchor_idx[valid_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_mask]
        self.denom = self.denom[valid_mask]
        self.max_radii2D = self.max_radii2D[valid_mask]

    def _prune_points(self, prune_mask: torch.Tensor) -> None:
        self._prune_optimizer(~prune_mask)

    def _cat_tensors_to_optimizer(
        self, extension_dict: Dict[str, torch.Tensor],
        new_anchor_idx: torch.Tensor,
    ) -> None:
        if not extension_dict:
            return
        any_name = next(iter(extension_dict))
        M = int(extension_dict[any_name].shape[0])
        if M == 0:
            return
        for name, extension in extension_dict.items():
            if name not in self.group_idx:
                continue
            g_idx = self.group_idx[name]
            group = self.optimizer.param_groups[g_idx]
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)

            new_data = torch.cat([old_param.data, extension], dim=0)
            new_param = torch.nn.Parameter(new_data.requires_grad_(True))
            if stored_state is not None:
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = torch.cat(
                        [stored_state["exp_avg"], torch.zeros_like(extension)], dim=0,
                    )
                if "exp_avg_sq" in stored_state:
                    stored_state["exp_avg_sq"] = torch.cat(
                        [stored_state["exp_avg_sq"], torch.zeros_like(extension)], dim=0,
                    )
                del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = stored_state
            group["params"][0] = new_param
            setattr(self, name, new_param)

        self.anchor_idx = torch.cat([self.anchor_idx, new_anchor_idx], dim=0)
        zeros1 = torch.zeros(M, 1, device=self.device)
        zerosN = torch.zeros(M, device=self.device)
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, zeros1], dim=0)
        self.xyz_gradient_accum_abs = torch.cat([self.xyz_gradient_accum_abs, zeros1], dim=0)
        self.denom = torch.cat([self.denom, zeros1], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, zerosN], dim=0)

    def _replace_tensor_to_optimizer(
        self, new_tensor: torch.Tensor, name: str,
    ) -> None:
        if name not in self.group_idx:
            return
        g_idx = self.group_idx[name]
        group = self.optimizer.param_groups[g_idx]
        old_param = group["params"][0]
        stored_state = self.optimizer.state.get(old_param, None)

        new_param = torch.nn.Parameter(new_tensor.detach().requires_grad_(True))
        if stored_state is not None:
            stored_state["exp_avg"] = torch.zeros_like(new_param)
            stored_state["exp_avg_sq"] = torch.zeros_like(new_param)
            del self.optimizer.state[old_param]
            self.optimizer.state[new_param] = stored_state
        group["params"][0] = new_param
        setattr(self, name, new_param)

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, float]:
        opac = self.aux_logit_opacities.sigmoid().reshape(-1)
        offset_mag = self.aux_xyz_offset.detach().norm(dim=-1)
        scales = self.aux_log_scales.detach().exp()
        return {
            "N_total": float(self.num_gaussians),
            "mean_opacity": float(opac.mean().item()),
            "median_opacity": float(opac.median().item()),
            "mean_xyz_offset": float(offset_mag.mean().item()),
            "max_xyz_offset": float(offset_mag.max().item()),
            "mean_max_scale": float(scales.max(dim=1).values.mean().item()),
        }


# ---------------------------------------------------------------------------
# Aux-state factory
# ---------------------------------------------------------------------------
def _init_guided_gaussian_state(
    *, pts_init: torch.Tensor, init_normals: Optional[torch.Tensor],
    n_cams: int, initial_n_gaussians_per_anchor: int,
    initial_opacity: float, initial_scale_factor: float,
    initialize_isotropic_gaussians: bool,
    gaussian_radius: float, scene_scale: float,
    k_neighbors: int, learn_normals: bool, learn_camera_pose: bool,
    use_exposure_compensation: bool, sh_degree: int,
    aux_lr_multiplier: float, cam_rot_lr: float, cam_trans_lr: float,
    spatial_lr_scale: float, scene_extent: float, device: torch.device,
    use_position_lr_schedule: bool = False,
    position_lr_init: float = 0.00016,
    position_lr_final: float = 1.6e-6,
    position_lr_delay_mult: float = 0.01,
    position_lr_delay_steps: int = 0,
    position_lr_max_steps: int = 30_000,
    use_sh_degree_warmup: bool = False,
    sh_degree_warmup_interval: int = 1000,
) -> GuidedGaussianAuxState:
    P = int(pts_init.shape[0])
    n_g = max(1, int(initial_n_gaussians_per_anchor))

    if initialize_isotropic_gaussians:
        _, scaling, rotation = get_isotropic_gaussian_parameters_from_point_cloud(pts_init)
    else:
        _, scaling, rotation = get_gaussian_parameters_from_point_cloud(
            points=pts_init, k_neighbors=k_neighbors,
            min_scale=gaussian_radius * scene_scale * 1e-3,
            max_scale=2.0 * gaussian_radius * scene_scale,
            means=pts_init, normals=None,
        )
    scaling = scaling * initial_scale_factor
    log_scales_per_anchor = torch.log(scaling.clamp_min(1e-8))
    if n_g > 1:
        log_scales_per_anchor = log_scales_per_anchor - np.log(n_g ** (1.0 / 3.0))
    quats_per_anchor = torch.nn.functional.normalize(rotation, dim=-1)

    aux_log_scales = log_scales_per_anchor.unsqueeze(1).expand(P, n_g, 3).reshape(-1, 3).contiguous()
    aux_quats = quats_per_anchor.unsqueeze(1).expand(P, n_g, 4).reshape(-1, 4).contiguous()
    aux_colors = torch.full((P * n_g, 3), 0.5, device=device)
    init_opac = torch.full((P * n_g,), float(initial_opacity), device=device)
    aux_logit_opacities = inverse_sigmoid(init_opac).unsqueeze(-1)

    if n_g > 1:
        aux_xyz_offset = (
            torch.randn(P * n_g, 3, device=device)
            * 0.5 * aux_log_scales.exp().mean(dim=-1, keepdim=True)
        )
    else:
        aux_xyz_offset = torch.zeros(P * n_g, 3, device=device)

    if sh_degree > 0:
        num_sh = (sh_degree + 1) ** 2 - 1
        aux_colors_sh = torch.zeros(P * n_g, num_sh, 3, device=device)
    else:
        num_sh = 0
        aux_colors_sh = None

    if learn_normals:
        aux_normals = torch.zeros(P * n_g, 4, device=device)
        if init_normals is not None:
            aux_normals[:, :3] = init_normals.unsqueeze(1).expand(P, n_g, 3).reshape(-1, 3)
    else:
        aux_normals = None

    aux_delta_xyz = torch.zeros(P, 3, device=device)

    if learn_camera_pose:
        aux_cam_quats_t = torch.zeros(n_cams, 4, device=device)
        aux_cam_quats_t[:, 0] = 1.0
        aux_cam_trans_t = torch.zeros(n_cams, 3, device=device)
    else:
        aux_cam_quats_t = None
        aux_cam_trans_t = None
    aux_exposure_coeffs_t = (
        torch.zeros(n_cams, 2, device=device) if use_exposure_compensation else None
    )

    def _p(t: Optional[torch.Tensor]) -> Optional[torch.nn.Parameter]:
        if t is None:
            return None
        return torch.nn.Parameter(t.detach().clone().requires_grad_(True))

    p_xyz_offset = _p(aux_xyz_offset)
    p_log_scales = _p(aux_log_scales)
    p_quats = _p(aux_quats)
    p_colors = _p(aux_colors)
    p_logit_opac = _p(aux_logit_opacities)
    p_normals = _p(aux_normals)
    p_colors_sh = _p(aux_colors_sh)
    p_delta_xyz = _p(aux_delta_xyz)
    p_cam_quats = _p(aux_cam_quats_t)
    p_cam_trans = _p(aux_cam_trans_t)
    p_exp = _p(aux_exposure_coeffs_t)

    anchor_idx = (
        torch.arange(P, device=device)
        .unsqueeze(1).expand(P, n_g).reshape(-1).contiguous().long()
    )

    groups, group_idx = _build_optimizer_groups(
        aux_xyz_offset=p_xyz_offset, aux_log_scales=p_log_scales,
        aux_quats=p_quats, aux_colors=p_colors,
        aux_logit_opacities=p_logit_opac,
        aux_normals=p_normals, aux_colors_sh=p_colors_sh,
        aux_delta_xyz=p_delta_xyz,
        aux_cam_quats=p_cam_quats, aux_cam_trans=p_cam_trans,
        aux_exposure_coeffs=p_exp,
        spatial_lr_scale=spatial_lr_scale,
        aux_lr_multiplier=aux_lr_multiplier,
        cam_rot_lr=cam_rot_lr, cam_trans_lr=cam_trans_lr,
    )
    optimizer = Adam(groups, lr=0.0, eps=1e-15)

    position_scheduler: Optional[Callable[[int], float]] = None
    if use_position_lr_schedule:
        scale = float(spatial_lr_scale) * float(aux_lr_multiplier)
        position_scheduler = _get_expon_lr_func(
            lr_init=float(position_lr_init) * scale,
            lr_final=float(position_lr_final) * scale,
            lr_delay_steps=int(position_lr_delay_steps),
            lr_delay_mult=float(position_lr_delay_mult),
            max_steps=int(position_lr_max_steps),
        )

    return GuidedGaussianAuxState(
        aux_xyz_offset=p_xyz_offset, aux_log_scales=p_log_scales,
        aux_quats=p_quats, aux_colors=p_colors,
        aux_logit_opacities=p_logit_opac,
        aux_normals=p_normals, aux_colors_sh=p_colors_sh,
        anchor_idx=anchor_idx,
        aux_delta_xyz=p_delta_xyz,
        aux_cam_quats=p_cam_quats, aux_cam_trans=p_cam_trans,
        aux_exposure_coeffs=p_exp,
        optimizer=optimizer, group_idx=group_idx,
        num_anchors=P, num_sh=num_sh, sh_degree=sh_degree,
        learn_normals=learn_normals,
        scene_extent=scene_extent, device=device,
        position_scheduler=position_scheduler,
        active_sh_degree=(0 if use_sh_degree_warmup else int(sh_degree)),
        sh_degree_warmup_interval=(
            int(sh_degree_warmup_interval) if use_sh_degree_warmup else 0
        ),
    )


class _GuidedRun:
    """Stateful engine backing :func:`surflo.inference.engine.guided_inference`.

    ``__init__`` performs all of the one-shot scene preparation. The public
    methods called by the ODE loop (in
    :func:`surflo.inference.engine.run_guided_flow`) each wrap one conceptual
    stage: :meth:`flow_velocity` (the raw flow model), :meth:`guide` (rendering
    guidance driving the densifiable Gaussian model), :meth:`euler_step`,
    :meth:`polish` and :meth:`build_result`. Densification / pruning /
    opacity-reset are folded into the inner optimisation step and stay out
    of sight of the top-level loop.
    """

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def __init__(self, params: dict):
        # Every ``guided_inference`` parameter becomes an attribute
        # of the same name (config is read as ``self.<param>`` throughout).
        self.__dict__.update(params)

        model = self.model
        device = model.device
        self.device = device
        self.estimate_normals = model.estimate_normals

        # Disabled unless the caller passed profile=True, in which case every
        # `phase()` block below is a bare `yield` (no sync, no timing).
        self.profiler = PhaseTimer(
            enabled=bool(getattr(self, "profile", False)), device=device,
        )

        assert 0 <= self.sh_degree <= 3
        assert self.initial_n_gaussians_per_anchor >= 1

        if self.use_monodepth_guidance:
            assert self.monodepths is not None
        if self.use_normal_guidance:
            assert self.normal_guidances is not None

        if self.use_biphase:
            self.num_steps = self.num_steps_phase_1 + self.num_steps_phase_2

        # ---- Confidence masks -------------------------------------------
        self.conf_quad = None
        if self.use_confidence:
            self.conf_quad = build_confidence_masks(
                self.batch, self.scene_idx,
                confidence_threshold=self.confidence_threshold,
                use_smooth_confidence_mask=self.use_smooth_confidence_mask,
                smooth_confidence_mask_min_value=self.smooth_confidence_mask_min_value,
                device=device,
            )
            if self.conf_quad is None:
                self.use_confidence = False

        self.log_freq = max(self.num_steps // 10, 1)

        # ---- Per-scene data ---------------------------------------------
        self.scene_tokens = [
            t[self.scene_idx:self.scene_idx + 1] if t is not None else None
            for t in self.batch["aggregated_tokens_list"]
        ]
        vggt_wp = self.batch.get("vggt_world_points")
        self.scene_wp = vggt_wp[self.scene_idx:self.scene_idx + 1] if vggt_wp is not None else None
        self.patch_start_idx = self.batch["patch_start_idx"]
        self.cull_radius = self.batch["cull_radius"][self.scene_idx] if "cull_radius" in self.batch else None

        intrinsics = self.batch["vggt_intrinsics"][self.scene_idx]
        extrinsics = self.batch["vggt_extrinsics"][self.scene_idx]

        rgb_images = self.batch.get("rgb_images")
        if rgb_images is not None:
            scene_images = rgb_images[self.scene_idx]
        else:
            scene_images = self.batch.get("images")
            if scene_images is not None:
                scene_images = scene_images[self.scene_idx]

        cameras = get_cameras_from_intrinsics_and_extrinsics(
            intrinsics=intrinsics, extrinsics=extrinsics,
            images=scene_images, data_device=device,
        )

        # ---- Optional uniform-world rescaling ---------------------------
        original_camera_extent = float(get_cameras_spatial_extent(cameras=cameras)["radius"])
        self.effective_camera_extent = original_camera_extent
        self.scene_scale = 1.0
        if self.target_camera_extent is not None and self.target_camera_extent > 0.0:
            if original_camera_extent <= 0.0:
                _log.warning(
                    f"target_camera_extent={self.target_camera_extent} but extent="
                    f"{original_camera_extent}; skipping rescaling."
                )
            else:
                self.scene_scale = float(self.target_camera_extent) / original_camera_extent
                extrinsics_scaled = extrinsics.clone()
                extrinsics_scaled[..., :3, 3] = extrinsics_scaled[..., :3, 3] * self.scene_scale
                cameras = get_cameras_from_intrinsics_and_extrinsics(
                    intrinsics=intrinsics, extrinsics=extrinsics_scaled,
                    images=scene_images, data_device=device,
                )
                self.effective_camera_extent = float(get_cameras_spatial_extent(cameras=cameras)["radius"])
                _log.info(
                    f"Scene rescaling: extent {original_camera_extent:.4f} -> "
                    f"{self.effective_camera_extent:.4f} (target={self.target_camera_extent}, "
                    f"scale={self.scene_scale:.4f})"
                )
        self.cameras = cameras

        if self.camera_indices is None:
            self.camera_indices = list(range(len(cameras)))
        self.gt_images = [cameras[ci].original_image.float() for ci in self.camera_indices]

        self.vggt_depths = None
        if self.use_vggt_depth_loss:
            self.vggt_depths = build_vggt_depths(
                self.batch, self.scene_idx, cameras, self.camera_indices, self.scene_scale,
            )

        self.scene_mean, self.scene_std = None, None
        self.cull_mean, self.cull_std = None, None
        if model.per_scene_normalize and self.scene_wp is not None:
            self.scene_mean, self.scene_std = model._compute_scene_stats(self.scene_wp)
            if model.renormalize_after_cull and self.cull_radius is not None:
                self.cull_mean, self.cull_std = self.scene_mean, self.scene_std
                self.scene_mean, self.scene_std = model._compute_post_cull_scene_stats(
                    self.scene_wp, self.cull_mean, self.cull_std, self.cull_radius,
                )

        self.img_cull_masks: Optional[List[torch.Tensor]] = None
        if self.cull_radius is not None:
            img_mask_mean = self.cull_mean if self.cull_mean is not None else (
                self.scene_mean if self.scene_mean is not None else model.spatial_mean
            )
            img_mask_std = self.cull_std if self.cull_std is not None else (
                self.scene_std if self.scene_std is not None else model.spatial_std
            )
            self.img_cull_masks = build_cull_masks(
                self.batch, self.scene_idx,
                cull_radius=self.cull_radius,
                img_mask_mean=img_mask_mean, img_mask_std=img_mask_std,
                camera_indices=self.camera_indices,
                gt_images=self.gt_images, vggt_depths=self.vggt_depths,
            )
            _log.info(f"Created {len(self.img_cull_masks)} cull masks with shape {self.img_cull_masks[0].shape}")

        if self.guidance_images is not None:
            self.gt_images, self.vggt_depths, self.img_cull_masks, self.conf_quad = upsample_supervision(
                guidance_images=self.guidance_images,
                cameras=cameras, camera_indices=self.camera_indices,
                gt_images=self.gt_images, vggt_depths=self.vggt_depths,
                img_cull_masks=self.img_cull_masks,
                use_confidence=self.use_confidence,
                use_smooth_confidence_mask=self.use_smooth_confidence_mask,
                conf_quad=self.conf_quad, device=device,
            )

        # ---- Per-camera caches built AFTER any hi-res override ----------
        self.cam_geo = CameraGeometryCache(cameras, device)

        self.conf_mask_per_cam: Optional[List[torch.Tensor]] = None
        self.inv_conf_mask_per_cam: Optional[List[torch.Tensor]] = None
        if self.use_confidence and self.conf_quad is not None:
            conf_mask, inv_conf_mask, _binary, _inv_binary = self.conf_quad
            self.conf_mask_per_cam = [conf_mask[ci].unsqueeze(0) for ci in range(conf_mask.shape[0])]
            self.inv_conf_mask_per_cam = [inv_conf_mask[ci].unsqueeze(0) for ci in range(inv_conf_mask.shape[0])]

        self.monodepths_per_cam: Optional[List[torch.Tensor]] = None
        if self.monodepths is not None:
            self.monodepths_per_cam = [self.monodepths[ci] for ci in range(self.monodepths.shape[0])]

        self.normal_guidance_per_cam: Optional[List[torch.Tensor]] = None
        self.normal_guidance_curv_per_cam: Optional[List[torch.Tensor]] = None
        if self.normal_guidances is not None:
            self.normal_guidance_per_cam = [
                self.normal_guidances[ci] for ci in range(self.normal_guidances.shape[0])
            ]
            # Curvature target is constant for the entire run -- compute once.
            if self.use_curvature_loss_for_normal_guidance:
                self.normal_guidance_curv_per_cam = [
                    normal_to_curvature(ng) for ng in self.normal_guidance_per_cam
                ]

        # ---- Pre-compute compressed tokens (one-shot) -------------------
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            self.compressed_tokens, _ = model.surface_net.get_compressed_tokens(
                self.scene_tokens, self.patch_start_idx,
                vggt_world_points=self.scene_wp,
                scene_mean=self.scene_mean, scene_std=self.scene_std,
                cull_radius=self.cull_radius, cull_mean=self.cull_mean, cull_std=self.cull_std,
            )
            self.compressed_camera_tokens = None
            if (model.surface_net.use_camera_tokens
                    and model.surface_net.encode_camera_tokens_separately):
                self.compressed_camera_tokens = model.surface_net.get_compressed_camera_tokens(
                    self.scene_tokens, self.patch_start_idx,
                )

        with torch.no_grad():
            x = model.sample_from_source_distribution(
                n_points=self.num_query_points, batch_size=1,
                vggt_world_points=self.scene_wp, cull_radius=self.cull_radius,
                scene_mean=self.scene_mean, scene_std=self.scene_std,
                cull_mean=self.cull_mean, cull_std=self.cull_std,
            ).squeeze(0)
        self.x = x
        self.P = x.shape[0]

        if self.use_biphase:
            timesteps = get_biphase_time_grid(
                self.num_steps_phase_1, self.num_steps_phase_2, self.phase_switch_frac,
            ).to(device)
        else:
            timesteps = model.t_sampler.get_time_grid(self.num_steps, device=device)
        if self.start_time > 0.0:
            _log.info(f"Starting time: {self.start_time}")
            timesteps = self.start_time + (1.0 - self.start_time) * timesteps
        self.timesteps = timesteps

        # ---- Pre-compute per-step lambda lookups ------------------------
        total_outer_steps = self.num_steps + max(0, self.n_additional_iterations) + 8
        self.lambda_depth_table = precompute_lambda_schedule(
            self.lambda_depth, self.lambda_depth_schedule_values,
            self.lambda_depth_schedule_steps, total_outer_steps,
        )
        self.lambda_mono_table = precompute_lambda_schedule(
            self.lambda_monodepth_guidance,
            self.lambda_monodepth_schedule_values, self.lambda_monodepth_schedule_steps,
            total_outer_steps,
        )
        self.lambda_norm_table = precompute_lambda_schedule(
            self.lambda_normal_guidance,
            self.lambda_normal_guidance_schedule_values, self.lambda_normal_guidance_schedule_steps,
            total_outer_steps,
        )
        self.lambda_curv_table = precompute_lambda_schedule(
            self.lambda_curvature_loss,
            self.lambda_curvature_loss_schedule_values, self.lambda_curvature_loss_schedule_steps,
            total_outer_steps,
        )

        self.latest_render_pkg: List[Optional[Dict[str, torch.Tensor]]] = [None]
        self.latest_render_pkgs: List[Dict[str, torch.Tensor]] = []

        # ---- ODE-loop state ---------------------------------------------
        self.intermediates: List[torch.Tensor] = []
        if self.save_intermediates:
            self.intermediates.append(x.detach().clone())
        self.render_losses: List[Optional[float]] = []
        self.has_started_guidance = False

        self.state: Optional[GuidedGaussianAuxState] = None
        self.aux_cam_quats = None
        self.aux_cam_trans = None
        self.last_delta_norm = None
        self.densify_summary: Dict[str, int] = {
            "n_clone": 0, "n_split": 0, "n_prune": 0, "n_resets": 0,
        }
        self.render_iter = 0
        self.view_sampler = ViewSampler()

    # ------------------------------------------------------------------
    # ODE driving
    # ------------------------------------------------------------------
    def ode_steps(self):
        """Yield one :class:`_OdeStep` per Euler step (time bookkeeping)."""
        for step_idx in range(self.num_steps):
            t_curr = self.timesteps[step_idx]
            t_next = self.timesteps[step_idx + 1]
            dt = t_next - t_curr
            true_t_curr = t_curr.clone()
            t_curr = torch.clamp_max(t_curr, 1 - 5e-3)
            yield _OdeStep(idx=step_idx, t_curr=t_curr, true_t_curr=true_t_curr, dt=dt)

    def flow_velocity(self, x: torch.Tensor, step: _OdeStep) -> torch.Tensor:
        """Raw flow-model velocity ``v(x, t)`` (no grad, bf16 autocast)."""
        model = self.model
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model.surface_net(
                x, step.true_t_curr, self.scene_tokens, self.patch_start_idx,
                vggt_world_points=self.scene_wp,
                compressed_tokens=self.compressed_tokens,
                compressed_camera_tokens=self.compressed_camera_tokens,
                scene_mean=self.scene_mean, scene_std=self.scene_std,
                cull_radius=self.cull_radius, cull_mean=self.cull_mean, cull_std=self.cull_std,
            )
            prediction = result[0] if isinstance(result, tuple) else result
            if model.prediction_mode == "target":
                velocity = model.path.target_to_velocity(
                    x_1=prediction.unsqueeze(0), x_t=x.unsqueeze(0), t=step.t_curr,
                ).squeeze(0)
            else:
                velocity = prediction
        return velocity.float()

    def guide(self, x: torch.Tensor, velocity: torch.Tensor, step: _OdeStep) -> torch.Tensor:
        """Bend ``velocity`` towards the rendering target (under the hood).

        Once inside the guidance window this spins up the densifiable
        Gaussian model and runs ``k_inner_loop`` optimisation steps (each of
        which may clone / split / prune / reset opacity), then replaces
        ``velocity`` with the one pointing at the guided target.
        """
        start_guidance = (step.true_t_curr >= self.render_guidance_start_frac
                          and not self.has_started_guidance)
        step.apply_guidance = (step.true_t_curr >= self.render_guidance_start_frac
                               and self.render_guidance_scale > 0.0)

        if start_guidance:
            self._init_aux_state(x, velocity, step.t_curr)
        if step.apply_guidance:
            velocity = self._run_inner_guidance(x, velocity, step)
        return velocity

    def euler_step(self, x: torch.Tensor, velocity: torch.Tensor, step: _OdeStep) -> torch.Tensor:
        """Take the Euler step and run the optional callback / logging."""
        model = self.model
        x = x + step.dt * velocity
        self.render_losses.append(step.render_loss)
        if self.save_intermediates:
            self.intermediates.append(x.detach().clone())

        if self.step_callback is not None:
            with torch.no_grad():
                cb_world = model.unlift_points_from_flow_space(
                    x.unsqueeze(0), self.scene_mean, self.scene_std,
                )
                if self.estimate_normals:
                    cb_pts, cb_nrm = model.get_points_normals_from_r6_points(
                        cb_world.squeeze(0)
                    )
                else:
                    cb_pts = cb_world.squeeze(0)[..., :3]
                    cb_nrm = None

                cb_aux: Optional[Dict[str, torch.Tensor]] = None
                if self.has_started_guidance:
                    state = self.state
                    cb_delta = state.aux_delta_xyz.detach()
                    if self.scene_scale != 1.0:
                        cb_delta = cb_delta / self.scene_scale
                    cb_pts = cb_pts + cb_delta

                    cb_offset = state.aux_xyz_offset.detach()
                    if self.scene_scale != 1.0:
                        cb_offset = cb_offset / self.scene_scale
                    cb_pts = cb_pts[state.anchor_idx] + cb_offset

                    cb_scales = state.aux_log_scales.detach().exp()
                    if self.scene_scale != 1.0:
                        cb_scales = cb_scales / self.scene_scale
                    cb_aux = {
                        "scales": cb_scales,
                        "quats": torch.nn.functional.normalize(
                            state.aux_quats.detach(), dim=-1,
                        ),
                        "opacities": state.aux_logit_opacities.detach().sigmoid().reshape(-1),
                        "colors": state.aux_colors.detach(),
                    }

                    if self.learn_normals and state.aux_normals is not None:
                        cb_nrm = convert_features_to_normals(
                            features=state.aux_normals.detach(),
                        )
                    elif cb_nrm is not None:
                        cb_nrm = cb_nrm[state.anchor_idx]

            self.step_callback(
                step_idx=step.idx, num_steps=self.num_steps,
                t_curr=float(step.true_t_curr.item()),
                points=cb_pts.detach(),
                normals=cb_nrm.detach() if cb_nrm is not None else None,
                aux_params=cb_aux,
                has_started_guidance=self.has_started_guidance,
                cameras=self.cameras,
            )

        if step.idx % self.log_freq == 0 or step.idx == self.num_steps - 1:
            # render_loss is a deferred 0-dim tensor; only sync it here (log cadence).
            loss_str = (
                f"{float(step.render_loss):.5f}" if step.render_loss is not None else "---"
            )
            delta_norm_str = (
                f"{self.last_delta_norm.item():.5f}"
                if (step.apply_guidance and self.last_delta_norm is not None) else "---"
            )
            _log.info(
                f"  step {step.idx:3d}/{self.num_steps}  t={step.true_t_curr.item():.3f}  "
                f"||v||={velocity.norm().item():.3f}  "
                f"render_loss={loss_str} delta_norm={delta_norm_str}"
            )
        return x

    # ------------------------------------------------------------------
    # Densification bookkeeping
    # ------------------------------------------------------------------
    def _lam(self, table: torch.Tensor, step: int) -> float:
        return float(table[min(step, table.shape[0] - 1)].item())

    def _maybe_log_diag(self, prefix: str) -> None:
        if not self.log_densify_diagnostics:
            return
        d = self.state.diagnostics()
        _log.info(
            f"[densify-diag] {prefix} N={int(d['N_total'])} "
            f"opac(mean={d['mean_opacity']:.3f}, med={d['median_opacity']:.3f}) "
            f"|xyz_off|(mean={d['mean_xyz_offset']:.4f}, max={d['max_xyz_offset']:.4f}) "
            f"max_scale_mean={d['mean_max_scale']:.4f}"
        )

    def _maybe_densify_and_reset(self, local_render_iter: int) -> None:
        state = self.state
        if local_render_iter >= self.densify_until_step:
            return
        if (self.densification_interval > 0
                and local_render_iter > self.densify_from_step
                and local_render_iter % self.densification_interval == 0):
            # Screen-size pruning only kicks in after the first opacity reset.
            if (self.densify_max_screen_size_after_first_reset is not None
                    and self.opacity_reset_interval > 0
                    and local_render_iter > self.opacity_reset_interval):
                size_threshold = self.densify_max_screen_size_after_first_reset
            else:
                size_threshold = None
            info = state.densify_and_prune(
                max_grad=self.densify_grad_threshold,
                min_opacity=self.densify_min_opacity,
                max_screen_size=size_threshold,
                percent_dense=self.densify_percent_dense,
                densify_split_n=self.densify_split_n,
                use_abs_grad=self.densify_use_abs_grad,
            )
            self.densify_summary["n_clone"] += info["n_clone"]
            self.densify_summary["n_split"] += info["n_split"]
            self.densify_summary["n_prune"] += info["n_prune"]
            if self.log_densify_diagnostics:
                lr_str = (
                    f" pos_lr={state.last_position_lr:.2e}"
                    if state.last_position_lr is not None else ""
                )
                _log.info(
                    f"[densify] iter={local_render_iter} "
                    f"clone={info['n_clone']} split={info['n_split']} "
                    f"prune={info['n_prune']} "
                    f"N: {info['N_before']} -> {info['N_after']} "
                    f"size_th={size_threshold}{lr_str}"
                )
                self._maybe_log_diag(f"after densify@{local_render_iter}")
        if (self.opacity_reset_interval > 0
                and local_render_iter > 0
                and local_render_iter % self.opacity_reset_interval == 0):
            state.reset_opacity(opacity_reset_value=self.opacity_reset_value)
            self.densify_summary["n_resets"] += 1
            if self.log_densify_diagnostics:
                _log.info(f"[densify] iter={local_render_iter} opacity_reset fired")
                self._maybe_log_diag(f"after reset@{local_render_iter}")

    # ------------------------------------------------------------------
    # Guidance internals
    # ------------------------------------------------------------------
    def _run_one_aux_step(
        self, pts_0: torch.Tensor, nrm: Optional[torch.Tensor], gating_step_idx: int,
    ) -> torch.Tensor:
        """One Adam step on the densifiable Gaussian model (+ densify/prune).

        Returns the render loss as a *detached 0-dim tensor* (not a Python float):
        this is called ``k_inner_loop`` times per guided step, so materialising the
        value here would force a device sync every inner iteration. Callers store
        the tensor and defer the single ``.item()`` sync to logging / result build.
        """
        state = self.state
        state.update_learning_rate(self.render_iter)
        if state.maybe_oneup_sh_degree(self.render_iter):
            if self.log_densify_diagnostics:
                _log.info(
                    f"[sh-warmup] iter={self.render_iter} "
                    f"active_sh_degree -> {state.active_sh_degree}"
                    f"/{state.max_sh_degree}"
                )

        gs, gs_nograd, nrm_per_gauss = state.make_gaussians(
            pts_0=pts_0, anchor_normals=nrm,
            decouple_normals=self.decouple_normals,
        )

        state.optimizer.zero_grad()
        # One randomly sampled view per inner iteration.
        view_idx = self.view_sampler.next(len(self.gt_images))
        masks_slice = (
            [self.img_cull_masks[view_idx]] if self.img_cull_masks is not None else None
        )
        depths_slice = (
            self.vggt_depths[view_idx:view_idx + 1] if self.vggt_depths is not None else None
        )
        loss = self._render_loss(
            gs, nrm_per_gauss,
            [self.gt_images[view_idx]],
            [self.camera_indices[view_idx]],
            masks_slice,
            step_idx=gating_step_idx,
            gs_nograd=gs_nograd,
            vggt_depths_local=depths_slice,
            exposure_coeffs=state.aux_exposure_coeffs,
        )

        loss.backward()
        if self.render_iter < self.densify_until_step:
            for _pkg in self.latest_render_pkgs:
                state.add_densification_stats(_pkg)
        state.optimizer.step()

        self._maybe_densify_and_reset(self.render_iter)
        self.render_iter += 1
        return loss.detach()

    def _render_loss(
        self, gs: Gaussians, normals, gt_imgs, cam_idxs, rgb_cull_masks, step_idx,
        gs_nograd: Optional[Gaussians] = None,
        vggt_depths_local: Optional[torch.Tensor] = None,
        exposure_coeffs: Optional[torch.Tensor] = None,
    ):
        """Differentiable multi-view render loss over the aux Gaussians."""
        device = self.device
        if vggt_depths_local is None:
            vggt_depths_local = self.vggt_depths
        self.latest_render_pkgs.clear()

        # Hoisted: lambda values are camera-independent.
        cur_lambda_depth = self._lam(self.lambda_depth_table, step_idx)
        cur_lambda_mono = self._lam(self.lambda_mono_table, step_idx)
        cur_lambda_norm = self._lam(self.lambda_norm_table, step_idx)
        cur_lambda_curv = self._lam(self.lambda_curv_table, step_idx)

        total_loss = torch.tensor(0.0, device=device)
        for i_iter, (ci, gt_img) in enumerate(zip(cam_idxs, gt_imgs)):
            geo = self.cam_geo[ci]

            if self.use_random_bg:
                bg_color = torch.rand(3, device=device)
                gt_img = fill_gt_with_random_bg(
                    gt_img,
                    rgb_cull_masks[i_iter] if rgb_cull_masks is not None else None,
                    bg_color,
                )
            else:
                bg_color = None

            if self.learn_camera_pose:
                gs_render = apply_camera_pose_to_gaussians(
                    gs, self.aux_cam_quats[ci], self.aux_cam_trans[ci],
                )
                gs_nograd_render = (
                    apply_camera_pose_to_gaussians(
                        gs_nograd, self.aux_cam_quats[ci], self.aux_cam_trans[ci],
                    ) if gs_nograd is not None else None
                )
            else:
                gs_render = gs
                gs_nograd_render = gs_nograd

            normal_loss_active = (
                self.use_normal_loss and step_idx >= self.start_normal_loss_at_step
            )
            # When the normal loss is active, render RGB and the learned normals
            # as a single 6-channel pass instead of rasterizing twice.
            # Equivalence with a separate RGB + normal pass relies on
            # stop_normal_geometry_grad = decouple_normals (below) and on
            # detaching the depth target -- changing either breaks the gradient
            # equivalence.
            if normal_loss_active:
                assert normals is not None
                rgb_bg = bg_color if bg_color is not None else torch.zeros(3, device=device)
                pkg = render_surflo(
                    viewpoint_camera=self.cameras[ci],
                    gaussians=gs_render,
                    bg_color=rgb_bg,
                    require_coord=False,
                    require_depth=True,
                    normals=normals,
                    normal_bg_color=torch.zeros(3, device=device),
                    stop_normal_geometry_grad=self.decouple_normals,
                )
            else:
                pkg = render_surflo(
                    viewpoint_camera=self.cameras[ci],
                    gaussians=gs_render,
                    bg_color=(
                        bg_color if bg_color is not None
                        else torch.zeros(3, device=gs_render.device)
                    ),
                    kernel_size=0.0,
                    scaling_modifier=1.0,
                    require_coord=False,
                    require_depth=True,
                )
            self.latest_render_pkgs.append(pkg)
            rendered = pkg["render"]

            if self.mask_bg and rgb_cull_masks is not None:
                m = rgb_cull_masks[i_iter]
                rendered = (rendered * m).detach() + (rendered * ~m)

            if self.use_rgb_loss:
                total_loss = total_loss + rgb_loss(
                    rendered, gt_img, lambda_rgb=self.lambda_rgb,
                    exposure_coeff=(
                        exposure_coeffs[ci] if self.use_exposure_compensation else None
                    ),
                )

            if self.use_dn_loss and step_idx >= self.start_dn_loss_at_step:
                total_loss = total_loss + dn_loss_cached(pkg, geo)

            if self.use_vggt_depth_loss and cur_lambda_depth > 0.0:
                depth_weight_mask = self.conf_mask_per_cam[ci] if self.use_confidence else None
                total_loss = total_loss + depth_loss(
                    pkg, vggt_depths_local[i_iter],
                    lambda_depth=cur_lambda_depth,
                    depth_weight_mask=depth_weight_mask,
                    scene_scale=self.scene_scale,
                )

            if self.use_mask_loss:
                rendered_mask = pkg["mask"]
                gt_mask = 1.0 - rgb_cull_masks[i_iter][0:1].float()
                _mask_loss = (rendered_mask - gt_mask).abs()
                total_loss = total_loss + _mask_loss.mean() * self.lambda_mask_loss

            if normal_loss_active:
                assert normals is not None
                # Reuse the fused pass: channels 3:5 are the learned normals
                # (geometry grad already stopped when decoupling); detach the
                # median depth used as the alignment target.
                median_depth = pkg["median_depth"]
                if self.decouple_normals:
                    median_depth = median_depth.detach()
                _normal_loss, _ = normal_alignment_loss_cached(
                    pkg["normal_image"], median_depth, geo,
                )
                total_loss = total_loss + _normal_loss

            if self.use_anisotropy_penalty:
                anisotropy_max_ratio = 5.0
                ratio = gs.scales.max(dim=1).values / gs.scales.min(dim=1).values
                total_loss = total_loss + self.lambda_anisotropy_penalty * (
                    torch.clamp_min(ratio, anisotropy_max_ratio) - anisotropy_max_ratio
                ).mean()

            if self.use_entropy_loss and step_idx >= self.start_entropy_loss_at_step:
                opacities = gs.opacities
                total_loss = total_loss + self.lambda_entropy_loss * (
                    -opacities * torch.log(opacities + 1e-6)
                    - (1.0 - opacities) * torch.log(1.0 - opacities + 1e-6)
                ).mean()

            if (self.use_monodepth_guidance and step_idx >= self.start_monodepth_guidance_at_step
                    and cur_lambda_mono > 0.0):
                rendered_depth_blend = (
                    pkg["median_depth"] * self.depth_ratio_for_monodepth_guidance
                    + pkg["expected_depth"] * (1.0 - self.depth_ratio_for_monodepth_guidance)
                )
                total_loss = total_loss + cur_lambda_mono * compute_depth_order_loss(
                    depth=rendered_depth_blend,
                    prior_depth=self.monodepths_per_cam[ci],
                    scene_extent=self.effective_camera_extent,
                )

            if (self.use_normal_guidance and step_idx >= self.start_normal_guidance_at_step
                    and cur_lambda_norm > 0.0):
                rendered_depth_blend = (
                    pkg["median_depth"] * self.depth_ratio_for_normal_guidance
                    + pkg["expected_depth"] * (1.0 - self.depth_ratio_for_normal_guidance)
                )
                depthblend_normal, _ = depth_to_normal_with_mask_cached(
                    geo, rendered_depth_blend,
                )
                ng = self.normal_guidance_per_cam[ci]
                normal_error_map_rendered = 1.0 - (pkg["normal"] * ng).sum(dim=0)
                normal_error_map_depthblend = 1.0 - (depthblend_normal * ng).sum(dim=0)
                total_loss = total_loss + cur_lambda_norm * (
                    normal_error_map_rendered + normal_error_map_depthblend
                ).mean()

                if self.use_curvature_loss_for_normal_guidance and cur_lambda_curv > 0.0:
                    curv_target = self.normal_guidance_curv_per_cam[ci]
                    total_loss = total_loss + cur_lambda_curv * (
                        (normal_to_curvature(pkg["normal"]) - curv_target).abs().mean()
                        + (normal_to_curvature(depthblend_normal) - curv_target).abs().mean()
                    )

        self.latest_render_pkg[0] = pkg
        return total_loss / len(cam_idxs)

    def _init_aux_state(self, x: torch.Tensor, velocity: torch.Tensor, t_curr: torch.Tensor) -> None:
        """One-shot creation of the densifiable Gaussian model + optimizer."""
        model = self.model
        self.has_started_guidance = True
        with torch.no_grad():
            x1 = model.path.velocity_to_target(
                velocity=velocity.unsqueeze(0), x_t=x.unsqueeze(0), t=t_curr,
            ).squeeze(0)
            x1_world = model.unlift_points_from_flow_space(
                x1.unsqueeze(0), self.scene_mean, self.scene_std,
            )
            if self.estimate_normals:
                init_pts, init_nrm = model.get_points_normals_from_r6_points(
                    x1_world.squeeze(0),
                )
            else:
                init_pts = x1_world.squeeze(0)[..., :3]
                init_nrm = None
            if self.scene_scale != 1.0:
                init_pts = init_pts * self.scene_scale

            spatial_lr_scale = float(
                get_cameras_spatial_extent(cameras=self.cameras)["radius"]
            )

            self.state = _init_guided_gaussian_state(
                pts_init=init_pts, init_normals=init_nrm,
                n_cams=len(self.cameras),
                initial_n_gaussians_per_anchor=self.initial_n_gaussians_per_anchor,
                initial_opacity=self.initial_opacity,
                initial_scale_factor=self.initial_scale_factor,
                initialize_isotropic_gaussians=self.initialize_isotropic_gaussians,
                gaussian_radius=self.gaussian_radius,
                scene_scale=self.scene_scale,
                k_neighbors=self.k_neighbors,
                learn_normals=self.learn_normals,
                learn_camera_pose=self.learn_camera_pose,
                use_exposure_compensation=self.use_exposure_compensation,
                sh_degree=self.sh_degree,
                aux_lr_multiplier=self.aux_lr_multiplier,
                cam_rot_lr=self.cam_rot_lr,
                cam_trans_lr=self.cam_trans_lr * spatial_lr_scale,
                spatial_lr_scale=spatial_lr_scale,
                scene_extent=self.effective_camera_extent,
                device=self.device,
                use_position_lr_schedule=self.use_position_lr_schedule,
                position_lr_init=self.position_lr_init,
                position_lr_final=self.position_lr_final,
                position_lr_delay_mult=self.position_lr_delay_mult,
                position_lr_delay_steps=self.position_lr_delay_steps,
                position_lr_max_steps=self.position_lr_max_steps,
                use_sh_degree_warmup=self.use_sh_degree_warmup,
                sh_degree_warmup_interval=self.sh_degree_warmup_interval,
            )
            self.aux_cam_quats = self.state.aux_cam_quats
            self.aux_cam_trans = self.state.aux_cam_trans

    def _run_inner_guidance(self, x: torch.Tensor, velocity: torch.Tensor, step: _OdeStep) -> torch.Tensor:
        """Optimise the densifiable Gaussians for ``k_inner_loop`` steps and
        return the velocity pointing at the resulting guided target ``x1``."""
        model = self.model
        state = self.state
        device = self.device
        P = self.P
        t_curr = step.t_curr

        with torch.no_grad():
            x1 = model.path.velocity_to_target(
                velocity=velocity.detach().unsqueeze(0),
                x_t=x.detach().unsqueeze(0), t=t_curr,
            ).squeeze(0)
            x1_world = model.unlift_points_from_flow_space(
                x1.unsqueeze(0), self.scene_mean, self.scene_std,
            )
            if self.estimate_normals:
                pts_0, nrm = model.get_points_normals_from_r6_points(x1_world.squeeze(0))
            else:
                pts_0 = x1_world.squeeze(0)[..., :3]
                nrm = None
            if self.scene_scale != 1.0:
                pts_0 = pts_0 * self.scene_scale

        for _ in range(self.k_inner_loop):
            step.render_loss = self._run_one_aux_step(pts_0, nrm, step.idx)

        with torch.no_grad():
            guided_x1_pts = pts_0 + state.aux_delta_xyz.detach()
            if self.scene_scale != 1.0:
                guided_x1_pts = guided_x1_pts / self.scene_scale

            if state.aux_normals is not None:
                nrm_per_g = convert_features_to_normals(
                    features=state.aux_normals.detach(),
                )
                per_anchor_sum = torch.zeros(P, 3, device=device)
                per_anchor_count = torch.zeros(P, 1, device=device)
                per_anchor_sum.index_add_(0, state.anchor_idx, nrm_per_g)
                per_anchor_count.index_add_(
                    0, state.anchor_idx, state.ones_for_anchor_count(),
                )
                guided_x1_nrm = per_anchor_sum / per_anchor_count.clamp_min(1.0)
                if nrm is not None:
                    empty = (per_anchor_count.squeeze(-1) == 0)
                    if empty.any():
                        guided_x1_nrm[empty] = nrm[empty]
                guided_x1_nrm = torch.nn.functional.normalize(guided_x1_nrm, dim=-1)
            else:
                guided_x1_nrm = nrm

            guided_x1_pts = guided_x1_pts.unsqueeze(0)
            if self.estimate_normals:
                guided_x1_nrm_in = (
                    guided_x1_nrm.unsqueeze(0) if guided_x1_nrm is not None else None
                )
                guided_x1_world = model.get_r6_points_from_points_normals(
                    guided_x1_pts, guided_x1_nrm_in,
                )
            else:
                guided_x1_world = guided_x1_pts
            guided_x1 = model.lift_points_to_flow_space(
                guided_x1_world, self.scene_mean, self.scene_std,
            )
            velocity = model.path.target_to_velocity(
                x_1=guided_x1, x_t=x.detach().unsqueeze(0), t=t_curr,
            ).squeeze(0)

            self.last_delta_norm = state.aux_delta_xyz.detach().norm()
            state.aux_delta_xyz.data.zero_()
            state.optimizer.state.pop(state.aux_delta_xyz, None)

        return velocity

    # ------------------------------------------------------------------
    # Optional Phase C: aux-only polish
    # ------------------------------------------------------------------
    def polish(self, x: torch.Tensor) -> None:
        if not (self.n_additional_iterations > 0 and self.has_started_guidance):
            return
        model = self.model
        state = self.state
        with torch.no_grad():
            x_world_extra = model.unlift_points_from_flow_space(
                x.unsqueeze(0), self.scene_mean, self.scene_std,
            )
            if self.estimate_normals:
                pts_0_extra, nrm_extra = model.get_points_normals_from_r6_points(
                    x_world_extra.squeeze(0),
                )
            else:
                pts_0_extra = x_world_extra.squeeze(0)[..., :3]
                nrm_extra = None
            if self.scene_scale != 1.0:
                pts_0_extra = pts_0_extra * self.scene_scale

            pts_0_extra = pts_0_extra + state.aux_delta_xyz.detach()
            state.aux_delta_xyz.data.zero_()
            state.aux_delta_xyz.requires_grad = False
            state.optimizer.state.pop(state.aux_delta_xyz, None)

        log_freq_extra = max(1, self.n_additional_iterations // 10)
        for i_extra in range(self.n_additional_iterations):
            extra_loss = self._run_one_aux_step(
                pts_0_extra, nrm_extra, self.num_steps + i_extra,
            )
            # extra_loss is a deferred 0-dim tensor; keep it deferred (append the
            # tensor) and only sync inside the throttled logging branch.
            if i_extra % log_freq_extra == 0 or i_extra == self.n_additional_iterations - 1:
                _log.info(
                    f"  extra {i_extra:3d}/{self.n_additional_iterations}  "
                    f"render_loss={float(extra_loss):.5f}  "
                    f"N_total={state.num_gaussians}"
                )
            self.render_losses.append(extra_loss)

    # ------------------------------------------------------------------
    # Final unlift + result dict
    # ------------------------------------------------------------------
    def build_result(self, x: torch.Tensor) -> Dict[str, object]:
        model = self.model
        device = self.device

        with torch.no_grad():
            x_final = model.unlift_points_from_flow_space(
                x.unsqueeze(0), self.scene_mean, self.scene_std,
            )
            if self.estimate_normals:
                anchor_pts, anchor_nrm = model.get_points_normals_from_r6_points(
                    x_final.squeeze(0),
                )
            else:
                anchor_pts = x_final.squeeze(0)[..., :3]
                anchor_nrm = None

            if self.has_started_guidance:
                state = self.state
                delta_world = state.aux_delta_xyz.detach()
                if self.scene_scale != 1.0:
                    delta_world = delta_world / self.scene_scale
                anchor_pts_world = anchor_pts + delta_world

                offset_world = state.aux_xyz_offset.detach()
                if self.scene_scale != 1.0:
                    offset_world = offset_world / self.scene_scale
                pts_final = anchor_pts_world[state.anchor_idx] + offset_world

                if self.learn_normals and state.aux_normals is not None:
                    aux_normals_final = convert_features_to_normals(
                        features=state.aux_normals.detach(),
                    )
                    nrm_final = aux_normals_final.clone()
                else:
                    aux_normals_final = None
                    nrm_final = anchor_nrm[state.anchor_idx] if anchor_nrm is not None else None

                anchor_idx_out = state.anchor_idx.detach().clone()
            else:
                pts_final = anchor_pts
                nrm_final = anchor_nrm
                aux_normals_final = None
                anchor_idx_out = torch.arange(
                    pts_final.shape[0], device=device, dtype=torch.long,
                )

        aux_cam_quats_out = aux_cam_trans_out = None
        if self.learn_camera_pose and self.has_started_guidance:
            with torch.no_grad():
                q_unit = torch.nn.functional.normalize(self.state.aux_cam_quats.detach(), dim=-1)
                cos_half = q_unit[:, 0].clamp(-1.0, 1.0)
                angles_deg = (2.0 * torch.acos(cos_half) * 180.0 / math.pi)
                cam_trans_world = self.state.aux_cam_trans.detach()
                if self.scene_scale != 1.0:
                    cam_trans_world = cam_trans_world / self.scene_scale
                _log.info(
                    f"Camera pose correction: rot mean={angles_deg.mean().item():.3f} deg, "
                    f"max={angles_deg.max().item():.3f} deg | "
                    f"trans mean={cam_trans_world.norm(dim=-1).mean().item():.4f}, "
                    f"max={cam_trans_world.norm(dim=-1).max().item():.4f}"
                )
            aux_cam_quats_out = q_unit
            aux_cam_trans_out = cam_trans_world

        aux_exposure_coeffs_out = None
        if self.use_exposure_compensation and self.has_started_guidance:
            with torch.no_grad():
                ec = self.state.aux_exposure_coeffs.detach()
                gains = torch.exp(ec[:, 0])
                biases = ec[:, 1]
                _log.info(
                    f"Exposure compensation: gain mean={gains.mean().item():.3f}, "
                    f"min={gains.min().item():.3f}, max={gains.max().item():.3f} | "
                    f"bias mean={biases.mean().item():.4f}, "
                    f"min={biases.min().item():.4f}, max={biases.max().item():.4f}"
                )
            aux_exposure_coeffs_out = ec

        if self.has_started_guidance:
            state = self.state
            _log.info(
                f"> Densification summary: "
                f"clones={self.densify_summary['n_clone']} "
                f"splits={self.densify_summary['n_split']} "
                f"prunes={self.densify_summary['n_prune']} "
                f"opacity_resets={self.densify_summary['n_resets']} "
                f"final N_total={state.num_gaussians}"
            )

            aux_scales_out = state.aux_log_scales.detach().exp()
            if self.scene_scale != 1.0:
                aux_scales_out = aux_scales_out / self.scene_scale
            aux_colors_out = state.aux_colors.detach()
            aux_quats_out = torch.nn.functional.normalize(state.aux_quats.detach(), dim=-1)
            aux_opacities_out = state.aux_logit_opacities.detach().sigmoid().reshape(-1)
            aux_colors_sh_out = (
                state.aux_colors_sh.detach() if self.sh_degree > 0 else None
            )
        else:
            aux_scales_out = torch.zeros(0, 3, device=device)
            aux_colors_out = torch.zeros(0, 3, device=device)
            aux_quats_out = torch.zeros(0, 4, device=device)
            aux_opacities_out = torch.zeros(0, device=device)
            aux_colors_sh_out = None

        # Realise the deferred per-step render losses to Python floats (one sync).
        render_losses_out = materialize_deferred_losses(self.render_losses)

        return {
            "points": pts_final.detach(),
            # Full per-Gaussian centers (never culled downstream). Same values as
            # ``points`` here; kept under a stable name so the API can filter
            # ``points`` for display while meshing still sees every Gaussian.
            "aux_means": pts_final.detach(),
            "normals": nrm_final.detach() if nrm_final is not None else None,
            "intermediates": (
                torch.stack(self.intermediates, dim=0)
                if self.save_intermediates and self.intermediates else None
            ),
            "render_losses": render_losses_out,
            "aux_colors": aux_colors_out,
            "aux_scales": aux_scales_out,
            "aux_quats": aux_quats_out,
            "aux_opacities": aux_opacities_out,
            "aux_normals": (
                aux_normals_final.detach() if aux_normals_final is not None else None
            ),
            "aux_colors_sh": aux_colors_sh_out,
            "aux_cam_quats": aux_cam_quats_out,
            "aux_cam_trans": aux_cam_trans_out,
            "aux_exposure_coeffs": aux_exposure_coeffs_out,
            "supervision_images": self.gt_images,
            "anchor_idx": anchor_idx_out,
            # {"enabled": bool, "seconds": {...}, "calls": {...}};
            # all-empty unless the run was built with profile=True.
            "timings": self.profiler.as_dict(),
        }
