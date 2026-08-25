"""Plain (unguided) Surflo inference.

Given a batch produced by :meth:`surflo.model.ffm.FFM.preprocess_images`
(or the eval dataloader), sample the source distribution, integrate the
flow-matching ODE, and return the oriented surface point cloud. This is a
thin, faithful wrapper around :meth:`FFM.batched_inference`; the heavy lifting
(source sampling, per-scene normalization, the ODE solve and the r6->points+
normals decoding) lives in the model.

No rendering, no Gaussians, no mesh — see :mod:`surflo.inference.guided`
for the rendering-guided variant.
"""
from __future__ import annotations
import logging

from typing import Any, Dict, List, Optional

import torch

from .engine import PhaseTimer, _OdeStep

_log = logging.getLogger(__name__)


def _select_scene_tokens(batch: dict, scene_idx: int):
    """Slice the cached per-scene VGGT tokens / world points out of a batch.

    Equivalent to re-running the (deterministic, frozen) VGGT backbone on the
    scene images, without the redundant pass.
    """
    aggregated_tokens_list = [
        t[scene_idx:scene_idx + 1] if t is not None else None
        for t in batch["aggregated_tokens_list"]
    ]
    patch_start_idx = batch["patch_start_idx"]
    world_points = batch.get("vggt_world_points")
    if world_points is not None:
        world_points = world_points[scene_idx:scene_idx + 1]
    return aggregated_tokens_list, patch_start_idx, world_points


@torch.no_grad()
def run_plain_inference(
    model,
    batch: dict,
    *,
    scene_idx: int = 0,
    num_steps: int = 100,
    num_query_points: int = 100_000,
    num_points_per_batch: int = 100_000,
    guidance_scale: float = 0.0,
    seed: int = 42,
    return_source: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    """Run the unguided flow and return the flowed (and optionally source) cloud.

    Args:
        model: a loaded :class:`FFM`.
        batch: output of ``model.preprocess_images(...)`` (or the eval loader).
        scene_idx: which scene in the (possibly batched) ``batch`` to run.
        num_steps: number of ODE integration steps.
        num_query_points: number of points to sample from the source
            distribution and flow.
        num_points_per_batch: chunk size for the point batches (memory knob;
            does not change results).
        guidance_scale: classifier-free guidance scale (0.0 = disabled).
        seed: seed for the per-scene source-noise generator (reproducible).
        return_source: also return the ``t=1`` source sample (for ``initial.ply``).

    Returns:
        dict with ``points`` / ``normals`` (the flowed surface, ``normals`` is
        ``None`` if the model does not estimate normals) and, when
        ``return_source``, ``source_points`` / ``source_normals``.
    """
    device = model.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    agg, psi, world_points = _select_scene_tokens(batch, scene_idx)
    cull_radius = None
    if "cull_radius" in batch:
        cull_radius = batch["cull_radius"][scene_idx]

    kwargs: Dict[str, Any] = dict(
        aggregated_tokens_list=agg,
        patch_start_idx=psi,
        world_points=world_points,
        num_steps=int(num_steps),
        num_query_points=int(num_query_points),
        num_points_per_batch=int(num_points_per_batch),
        cull_radius=cull_radius,
        guidance_scale=float(guidance_scale),
        return_intermediates=return_source,
        generator=gen,
    )

    out = model.batched_inference(**kwargs)

    # Unpack (points[, normals]) and (intermediates over steps, when requested).
    if model.estimate_normals:
        pts, nrm = out
    else:
        pts, nrm = out, None

    if return_source:
        # intermediates: (num_steps + 1, P, 3); [0] = source sample, [-1] = final.
        source_points = pts[0].detach().float()
        final_points = pts[-1].detach().float()
        source_normals = nrm[0].detach().float() if nrm is not None else None
        final_normals = nrm[-1].detach().float() if nrm is not None else None
    else:
        source_points = source_normals = None
        final_points = pts.detach().float()
        final_normals = nrm.detach().float() if nrm is not None else None

    return {
        "points": final_points,
        "normals": final_normals,
        "source_points": source_points,
        "source_normals": source_normals,
    }


class _PlainRun:
    """Minimal guidance-free runner backing
    :func:`surflo.inference.engine.plain_inference`.

    The unguided counterpart to :class:`surflo.inference.guided._GuidedRun`.
    ``__init__`` does only the one-shot setup the plain flow actually needs --
    per-scene normalisation stats, compressed VGGT tokens, the time grid and a
    seeded source generator -- so (unlike the guided runner) it needs no
    cameras, GT images, masks or Gaussians. The driver
    (:func:`surflo.inference.engine.run_plain_flow`) iterates :meth:`chunks`
    (memory knob) and, per chunk, :meth:`ode_steps` / :meth:`flow_velocity` /
    :meth:`euler_step`, then :meth:`finish_chunk`; :meth:`build_result` returns
    a dict with the same keys as the guided runner (``aux_*`` empty / ``None``).

    The scene-slicing, stats, token-compression, ``ode_steps`` and
    ``flow_velocity`` logic are deliberately kept identical to
    :class:`~surflo.inference.guided._GuidedRun` so the unguided velocities
    match the guided ones step-for-step. Keep them in sync.
    """

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def __init__(self, params: dict):
        # Every ``plain_inference`` parameter becomes an attribute of the same
        # name (mirrors ``_GuidedRun``).
        self.__dict__.update(params)

        model = self.model
        device = model.device
        self.device = device
        self.estimate_normals = model.estimate_normals

        # Inert unless the caller passed profile=True (see :class:`PhaseTimer`).
        self.profiler = PhaseTimer(
            enabled=bool(getattr(self, "profile", False)), device=device,
        )

        # ---- Extract single-scene data (same slicing as _GuidedRun) --
        self.scene_tokens = [
            t[self.scene_idx:self.scene_idx + 1] if t is not None else None
            for t in self.batch["aggregated_tokens_list"]
        ]
        vggt_wp = self.batch.get("vggt_world_points")
        self.scene_wp = vggt_wp[self.scene_idx:self.scene_idx + 1] if vggt_wp is not None else None
        self.patch_start_idx = self.batch["patch_start_idx"]
        self.cull_radius = self.batch["cull_radius"][self.scene_idx] if "cull_radius" in self.batch else None

        # ---- Scene normalisation stats (same as _GuidedRun) ---------
        self.scene_mean, self.scene_std = None, None
        self.cull_mean, self.cull_std = None, None
        if model.per_scene_normalize and self.scene_wp is not None:
            self.scene_mean, self.scene_std = model._compute_scene_stats(self.scene_wp)
            if model.renormalize_after_cull and self.cull_radius is not None:
                self.cull_mean, self.cull_std = self.scene_mean, self.scene_std
                self.scene_mean, self.scene_std = model._compute_post_cull_scene_stats(
                    self.scene_wp, self.cull_mean, self.cull_std, self.cull_radius,
                )

        # ---- Pre-compute compressed tokens (once, reused every step) -------
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

        # ---- Time grid (plain: no biphase) ---------------------------------
        timesteps = model.t_sampler.get_time_grid(self.num_steps, device=device)
        if self.start_time > 0.0:
            _log.info(f"Starting time: {self.start_time}")
            timesteps = self.start_time + (1.0 - self.start_time) * timesteps
        self.timesteps = timesteps

        # ---- Seeded source generator (reproducible) ------------------------
        self.generator = torch.Generator(device=device).manual_seed(int(self.seed))

        # ---- Result accumulators (concatenated across chunks) --------------
        self.x: Optional[torch.Tensor] = None
        self._pts_chunks: List[torch.Tensor] = []
        self._nrm_chunks: List[torch.Tensor] = []
        self._inter_chunks: List[torch.Tensor] = []
        self._chunk_inter: Optional[List[torch.Tensor]] = None
        # World-space source cloud (opt-in via ``return_source``, for initial.ply).
        self._src_pts_chunks: List[torch.Tensor] = []
        self._src_nrm_chunks: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # ODE driving
    # ------------------------------------------------------------------
    def chunks(self):
        """Yield once per point-batch, (re)seeding ``self.x`` with a fresh source
        sample drawn from the shared seeded generator (memory knob)."""
        n = int(self.num_query_points)
        step = int(self.num_points_per_batch) if self.num_points_per_batch else n
        step = max(1, step)
        for start in range(0, n, step):
            chunk = min(step, n - start)
            with torch.no_grad():
                self.x = self.model.sample_from_source_distribution(
                    n_points=chunk, batch_size=1,
                    vggt_world_points=self.scene_wp, cull_radius=self.cull_radius,
                    scene_mean=self.scene_mean, scene_std=self.scene_std,
                    cull_mean=self.cull_mean, cull_std=self.cull_std,
                    generator=self.generator,
                ).squeeze(0)
                if self.return_source:
                    s_pts, s_nrm = self._decode(self.x)
                    self._src_pts_chunks.append(s_pts.detach())
                    if s_nrm is not None:
                        self._src_nrm_chunks.append(s_nrm.detach())
            self._chunk_inter = (
                [self.x.detach().clone()] if self.save_intermediates else None
            )
            yield start, chunk

    def ode_steps(self):
        """Yield one :class:`_OdeStep` per Euler step (identical time
        bookkeeping to :meth:`_GuidedRun.ode_steps`)."""
        for step_idx in range(self.num_steps):
            t_curr = self.timesteps[step_idx]
            t_next = self.timesteps[step_idx + 1]
            dt = t_next - t_curr
            true_t_curr = t_curr.clone()
            t_curr = torch.clamp_max(t_curr, 1 - 5e-3)
            yield _OdeStep(idx=step_idx, t_curr=t_curr, true_t_curr=true_t_curr, dt=dt)

    def flow_velocity(self, x: torch.Tensor, step: _OdeStep) -> torch.Tensor:
        """Raw flow-model velocity ``v(x, t)`` (identical to
        :meth:`_GuidedRun.flow_velocity`)."""
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

    def euler_step(self, x: torch.Tensor, velocity: torch.Tensor, step: _OdeStep) -> torch.Tensor:
        """Take the Euler step (no guidance, no callback)."""
        x = x + step.dt * velocity
        if self.save_intermediates:
            self._chunk_inter.append(x.detach().clone())
        return x

    def _decode(self, x_flow: torch.Tensor):
        """Unlift a flow-space cloud to world space and split into points
        (+ normals). Shared by the source (``chunks``) and final
        (``finish_chunk``) decode paths."""
        model = self.model
        x_world = model.unlift_points_from_flow_space(
            x_flow.unsqueeze(0), self.scene_mean, self.scene_std,
        )
        if self.estimate_normals:
            pts, nrm = model.get_points_normals_from_r6_points(x_world.squeeze(0))
        else:
            pts = x_world.squeeze(0)[..., :3]
            nrm = None
        return pts, nrm

    def finish_chunk(self, x: torch.Tensor) -> None:
        """Unlift the flowed chunk to world space, decode normals, accumulate."""
        with torch.no_grad():
            pts, nrm = self._decode(x)
        self._pts_chunks.append(pts.detach())
        if nrm is not None:
            self._nrm_chunks.append(nrm.detach())
        if self.save_intermediates and self._chunk_inter is not None:
            self._inter_chunks.append(torch.stack(self._chunk_inter, dim=0))

    # ------------------------------------------------------------------
    # Final result dict (same keys as the guided runners)
    # ------------------------------------------------------------------
    def build_result(self) -> Dict[str, object]:
        device = self.device
        pts_final = torch.cat(self._pts_chunks, dim=0)
        nrm_final = torch.cat(self._nrm_chunks, dim=0) if self._nrm_chunks else None
        intermediates = (
            torch.cat(self._inter_chunks, dim=1)
            if (self.save_intermediates and self._inter_chunks) else None
        )
        result = {
            "points": pts_final.detach(),
            "normals": nrm_final.detach() if nrm_final is not None else None,
            "intermediates": intermediates,
            "render_losses": [],
            "aux_colors": torch.zeros(0, 3, device=device),
            "aux_scales": torch.zeros(0, 3, device=device),
            "aux_quats": torch.zeros(0, 4, device=device),
            "aux_opacities": torch.zeros(0, device=device),
            "aux_normals": None,
            "aux_cam_quats": None,
            "aux_cam_trans": None,
            "aux_exposure_coeffs": None,
            "supervision_images": None,
            # See _GuidedRun.build_result; empty unless profile=True.
            "timings": self.profiler.as_dict(),
        }
        # Opt-in, additive: keep the default schema uniform with the guided
        # variants, only exposing the world-space source when explicitly asked.
        if self.return_source:
            src_pts = torch.cat(self._src_pts_chunks, dim=0) if self._src_pts_chunks else None
            src_nrm = torch.cat(self._src_nrm_chunks, dim=0) if self._src_nrm_chunks else None
            result["source_points"] = src_pts.detach() if src_pts is not None else None
            result["source_normals"] = src_nrm.detach() if src_nrm is not None else None
        return result
