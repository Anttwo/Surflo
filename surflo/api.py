"""High-level Python API for playing with the Surflo model.

These entry points cover the common "load it and poke at it" workflow, and
produce the same results as the CLI at ``scripts/infer.py``:

  1. Load a model + checkpoint            -> :meth:`Surflo.from_checkpoint`
  2. Encode images -> preprocessed data    -> :meth:`Surflo.encode` (-> SceneState)
  3. Query the flow velocity ``v(x, t)``   -> :meth:`SceneState.velocity`
  4. Solve the ODE (plain / guided) -> :meth:`SceneState.reconstruct`
  5. Extract a mesh (guided results)       -> :meth:`SceneState.extract_mesh`
  6. TSDF-color points / mesh vertices     -> :meth:`SceneState.color_points`
                                              :meth:`SceneState.color_mesh`

Typical use::

    from surflo import Surflo, save_ply

    surflo = Surflo.from_checkpoint("surflo_v0.pt")
    scene = surflo.encode("path/to/images")            # a folder, paths, or tensors

    x = scene.sample_source(50_000, seed=0)            # flow-space source cloud
    v = scene.velocity(x, t=0.5)                        # flow-space velocity

    result = scene.reconstruct(mode="plain", num_query_points=100_000)
    save_ply(result, "surface.ply")
"""
from __future__ import annotations

import inspect
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .data.image_folder import build_image_folder_batch, list_images_in_folder
from .data.utils import load_and_preprocess_images
from .inference.engine import (
    guided_inference,
    plain_inference,
)
from .model.loader import load_model
from .utils.io import (
    normals_to_rgb_uint8,
    uint8_colors_from_floats,
    write_mesh_ply,
    write_points_ply,
)

_log = logging.getLogger(__name__)

# Bundled single-source-of-truth model config (configs/model/surflo.yaml).
_DEFAULT_MODEL_CFG = (
    Path(__file__).resolve().parents[1] / "configs" / "model" / "surflo.yaml"
)

# ODE-solve entry points, keyed by the ``mode`` string ``reconstruct`` accepts.
_ENGINE_FNS = {
    "plain": plain_inference,
    "guided": guided_inference,
}

# Monodepth-expert defaults (mirrors ``configs/expert/default.yaml``
# and :class:`surflo.inference.expert.MonodepthExpert`).
_EXPERT_DEFAULTS = {
    "enabled": True,
    "monodepth_id": "depth-anything/da3mono-large",
    "pred_res": 1596,
    "process_res_method": "upper_bound_resize",
    "conf_threshold_for_normal_guidance": 2.0,
}

# Loaded DA3 experts, cached by ``(model_id, device)`` so back-to-back scenes /
# reconstruct calls reuse the weights (mirrors ``scripts/infer.py``'s global
# ``_MONO_EXPERT_CACHE``).
_MONODEPTH_EXPERT_CACHE: Dict[Tuple[str, str], Any] = {}

# Wrapping-mesh extraction defaults (mirrors ``configs/mesh/default.yaml`` and
# :func:`surflo.extraction.occupancy.wrapping.pivot_extraction_with_binary_search`).
_MESH_DEFAULTS = {
    "kernel_size": 0.0,
    "sdf_isosurface_value": 0.0,
    "use_regular_pivots": True,
    "std_factor": 3.33,
    "n_pivots": 9,
    "use_smallest_axis_as_normal": True,
    "n_points_per_sdf_evaluation": 20_000_000,
    "use_valid_mask": True,
    "filter_large_edges": True,
    "collapse_large_edges": False,
    "mtet_on_cpu": False,
    "n_binary_steps": 10,
    # Delaunay backend for the pivot tetrahedralization ("geodel" | "scipy").
    "delaunay_method": "geodel",
    # Extract against the per-view corrected cameras the guided flow learned.
    "use_refined_cameras": True,
}


def _expert_cfg_to_dict(expert_cfg: Optional[Union[dict, DictConfig]]) -> dict:
    """Normalise an expert config (``DictConfig`` / dict / ``None``) to a dict."""
    if expert_cfg is None:
        return {}
    if isinstance(expert_cfg, DictConfig):
        return dict(OmegaConf.to_container(expert_cfg, resolve=True))
    return dict(expert_cfg)


def _resolve_expert_settings(cfg: dict) -> dict:
    """Overlay a (partial) expert config on top of :data:`_EXPERT_DEFAULTS`."""
    settings = dict(_EXPERT_DEFAULTS)
    settings.update({k: v for k, v in cfg.items() if k in _EXPERT_DEFAULTS})
    return settings


def _get_monodepth_expert(model_id: str, device: Union[str, torch.device]):
    """Lazily load (and cache) a :class:`MonodepthExpert` for ``(model_id, device)``."""
    key = (str(model_id), str(device))
    expert = _MONODEPTH_EXPERT_CACHE.get(key)
    if expert is None:
        # Imported here so plain / no-expert runs never touch DepthAnything3.
        from .inference.expert import MonodepthExpert

        _log.info(f"[api] loading monodepth expert ({model_id}) on {device}...")
        expert = MonodepthExpert(model_id=str(model_id), device=device)
        _MONODEPTH_EXPERT_CACHE[key] = expert
    return expert


def set_global_seeds(seed: int) -> None:
    """Seed Python / NumPy / torch RNGs, as ``scripts/infer.py`` does.

    The guided variants draw their source cloud and random backgrounds from the
    *global* torch RNG, so seeding here is what makes guided runs reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _filter_kwargs_to_signature(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs the target function does not accept (logging them once).

    Lets a Hydra preset block be passed straight through even if it carries
    knobs a given engine function does not expose.
    """
    valid = set(inspect.signature(fn).parameters.keys())
    unknown = sorted(set(kwargs) - valid)
    if unknown:
        _log.info(
            f"[api] dropping {len(unknown)} kwarg(s) not accepted by "
            f"{fn.__name__}: {unknown}"
        )
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return kwargs


@torch.no_grad()
def _finalize_guided_result(
    result: Dict[str, Any], threshold: Optional[float],
) -> Dict[str, Any]:
    """Expose an opacity-filtered *display* cloud while keeping every Gaussian.

    Unlike a *hard* cull, this never removes any Gaussian: the full parameter
    set (``aux_means`` / ``aux_scales`` / ``aux_quats`` / ``aux_opacities`` /
    ``aux_colors`` / ``aux_normals`` / ``aux_colors_sh``) is left intact so
    meshing / coloring see all Gaussians.
    It only (re)writes the display keys ``points`` / ``normals`` / ``colors`` to
    the ``aux_opacities >= threshold`` subset -- the equivalent of the CLI's
    ``point_cloud_normals.ply`` cloud.

    ``points`` is drawn from ``aux_means`` (the full centers), ``normals`` from
    the per-point ``normals`` map, and ``colors`` from the learned per-Gaussian
    albedo ``aux_colors``. No-op for a plain result (no ``aux_means``); when
    ``threshold`` is ``None`` / ``<= 0`` the whole cloud is kept (all Gaussians).
    """
    aux_means = result.get("aux_means")
    if not isinstance(aux_means, torch.Tensor):
        # Plain / no-Gaussians result: nothing to filter, no display cloud to build.
        return result

    full_means = aux_means.detach()
    full_normals = result.get("normals")
    full_colors = result.get("aux_colors")
    n = int(full_means.shape[0])

    opacities = result.get("aux_opacities")
    if (
        threshold is not None and float(threshold) > 0.0
        and isinstance(opacities, torch.Tensor)
        and opacities.detach().reshape(-1).numel() == n
    ):
        keep = opacities.detach().float().reshape(-1) >= float(threshold)
        _log.info(
            f"[api] opacity_threshold={threshold}: display cloud keeps "
            f"{int(keep.sum().item())}/{n} points (all {n} Gaussians retained)."
        )
    else:
        keep = torch.ones(n, dtype=torch.bool, device=full_means.device)

    out = dict(result)
    out["points"] = full_means[keep]
    out["normals"] = (
        full_normals[keep] if isinstance(full_normals, torch.Tensor) else None
    )
    out["colors"] = (
        full_colors[keep] if isinstance(full_colors, torch.Tensor) else None
    )
    return out


class Surflo:
    """Facade around a loaded :class:`~surflo.model.ffm.FFM`.

    Construct via :meth:`from_checkpoint` (Point 1), then :meth:`encode` a set of
    images into a :class:`SceneState` (Point 2).
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @property
    def device(self) -> torch.device:
        return self.model.device

    # ------------------------------------------------------------------
    # Point 1 -- model loading
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Optional[str] = None,
        model_cfg: Optional[Union[str, Path, dict, DictConfig]] = None,
        device: Union[str, torch.device] = "cuda",
        use_ema: bool = True,
    ) -> "Surflo":
        """Instantiate the model and load a checkpoint.

        Args:
            ckpt_path: path to a Surflo checkpoint (the released one is
                ``surflo_v0.pt``). If ``None``, an untrained model is returned
                (VGGT backbone still loaded from the hub).
            model_cfg: model architecture config, as a ``DictConfig`` / dict /
                path to a YAML. If ``None`` (default), the bundled
                ``configs/model/surflo.yaml`` (the released architecture) is used.
            device: device to place the model on.
            use_ema: load EMA weights (default, matches evaluation).
        """
        cfg = cls._resolve_model_cfg(model_cfg)
        model = load_model(cfg, ckpt_path=ckpt_path, device=device, use_ema=use_ema)
        return cls(model)

    @staticmethod
    def _resolve_model_cfg(
        model_cfg: Optional[Union[str, Path, dict, DictConfig]],
    ) -> DictConfig:
        if model_cfg is None:
            if not _DEFAULT_MODEL_CFG.is_file():
                raise FileNotFoundError(
                    f"Default model config not found at {_DEFAULT_MODEL_CFG}. "
                    "Pass model_cfg=... explicitly."
                )
            _log.info(f"Using default model config: {_DEFAULT_MODEL_CFG}")
            return OmegaConf.load(str(_DEFAULT_MODEL_CFG))
        if isinstance(model_cfg, DictConfig):
            return model_cfg
        if isinstance(model_cfg, (str, Path)):
            return OmegaConf.load(str(model_cfg))
        if isinstance(model_cfg, dict):
            return OmegaConf.create(model_cfg)
        raise TypeError(
            f"model_cfg must be None, a path, dict or DictConfig; got {type(model_cfg)}."
        )

    # ------------------------------------------------------------------
    # Point 2 -- image encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(
        self,
        images: Union[str, Path, List[Union[str, Path, torch.Tensor]], torch.Tensor],
        *,
        scene_idx: int = 0,
        target_size: int = 518,
        cull_radius: Optional[float] = 10.0,
        n_images: Optional[int] = None,
        image_sampling: str = "uniform",
        image_seed: int = 42,
    ) -> "SceneState":
        """Run the compressor over a scene's images and cache its global state.

        Accepts any of:
          * a folder path (str / ``Path``) of JPG/PNG images;
          * a list of image file paths;
          * a list of ``(3, H, W)`` tensors, or a stacked ``(N, 3, H, W)`` /
            ``(B, N, 3, H, W)`` tensor (values in ``[0, 1]``).

        ``cull_radius`` defaults to ``10.0`` to match ``configs/infer.yaml``'s
        ``source.cull_radius``; this is **required** for guided modes whose preset
        enables ``use_mask_loss`` (the mask loss compares against the cull masks).
        Pass ``cull_radius=None`` to disable spatial culling.

        Returns a :class:`SceneState` bundling the preprocessed VGGT data
        (resized images, depth / confidence maps, camera params, world points)
        together with the cached per-scene normalisation stats and compressed
        global tokens.
        """
        batch = self._build_batch(
            images,
            target_size=target_size,
            cull_radius=cull_radius,
            n_images=n_images,
            image_sampling=image_sampling,
            image_seed=image_seed,
        )
        return SceneState(self.model, batch, scene_idx=scene_idx)

    def _build_batch(
        self,
        images,
        *,
        target_size: int,
        cull_radius: Optional[float],
        n_images: Optional[int],
        image_sampling: str,
        image_seed: int,
    ) -> dict:
        model = self.model
        device = model.device

        # -- Folder path -> reuse the folder loader (handles sampling) ----------
        if isinstance(images, (str, Path)):
            folder = str(images)
            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"images={folder!r} is not a directory. Pass a folder, a list "
                    "of image paths, or a list/stack of image tensors."
                )
            n = n_images if n_images is not None else len(list_images_in_folder(folder))
            batch, _ = build_image_folder_batch(
                model,
                folder=folder,
                n_images=int(n),
                sampling=image_sampling,
                seed=int(image_seed),
                target_size=int(target_size),
                cull_radius=cull_radius,
                device=device,
            )
            return batch

        # -- List of paths / tensors, or a stacked tensor ----------------------
        img_tensor = self._to_image_tensor(images, target_size=target_size, device=device)
        cr = (
            float(cull_radius)
            if cull_radius is not None and float(cull_radius) > 0.0
            else None
        )
        batch = model.preprocess_images(images=img_tensor, cull_radius=cr)
        batch["seq_name"] = ["custom_scene"]
        return batch

    @staticmethod
    def _to_image_tensor(images, *, target_size: int, device) -> torch.Tensor:
        # Already a stacked tensor: (N, 3, H, W) or (B, N, 3, H, W).
        if isinstance(images, torch.Tensor):
            return images.to(device)
        if isinstance(images, (list, tuple)) and len(images) > 0:
            first = images[0]
            # List of file paths -> VGGT loader (resize / letterbox to target_size).
            if isinstance(first, (str, Path)):
                paths = [str(p) for p in images]
                return load_and_preprocess_images(
                    paths, mode="no_stretch", target_size=int(target_size),
                    rotate_portrait=True,  # inference: correct the landscape bias
                ).to(device)
            # List of (3, H, W) tensors -> stack to (N, 3, H, W).
            if isinstance(first, torch.Tensor):
                return torch.stack(list(images), dim=0).to(device)
        raise ValueError(
            "images must be a folder path, a list of image paths, a list of "
            "(3, H, W) tensors, or a stacked (N, 3, H, W) / (B, N, 3, H, W) tensor."
        )

    # ------------------------------------------------------------------
    # One-call reconstruction (mirrors scripts/infer.py end-to-end)
    # ------------------------------------------------------------------
    def reconstruct(
        self,
        images: Union[str, Path, List[Union[str, Path, torch.Tensor]], torch.Tensor],
        mode: str = "plain",
        *,
        config_block: Optional[Union[dict, DictConfig]] = None,
        expert_cfg: Optional[Union[dict, DictConfig]] = None,
        opacity_threshold: Optional[float] = 0.1,
        seed: int = 42,
        n_images: Optional[int] = None,
        sampling: str = "uniform",
        target_size: int = 518,
        cull_radius: Optional[float] = 10.0,
        scene_idx: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """End-to-end reconstruction from images -- the one-call equivalent of
        ``python scripts/infer.py mode=<mode> ...``.

        This is the batteries-included path: it seeds the RNGs, encodes the
        images (with the CLI-default ``cull_radius=10.0`` so guided mask losses
        have their cull masks), runs the DepthAnything-3 expert when the preset
        asks for it (``expert_cfg``), and solves the ODE with the correct
        grad / autocast context -- returning the engine result dict (feed it to
        :func:`save_ply`).

        For finer control (querying velocities, reusing one encoding across many
        solves), use :meth:`encode` + :meth:`SceneState.reconstruct` directly.

        Args:
            images: folder path, list of image paths, or image tensor(s) --
                anything :meth:`encode` accepts.
            mode: ``"plain"`` / ``"guided"``.
            config_block: resolved preset block for this mode (e.g. a
                Hydra-composed ``cfg.guided``, or a dict).
            expert_cfg: expert settings (``cfg.expert`` / dict / ``{}``); required
                only for presets that enable monodepth / normal guidance.
            opacity_threshold: opacity cutoff for the *display* cloud only
                (``points`` / ``normals`` / ``colors``); the full Gaussian set
                (``aux_*``) is always kept. Default ``0.1``; ``None`` keeps every
                point. No-op for ``plain``. See :meth:`SceneState.reconstruct`.
            seed: global RNG seed (guided source cloud + random backgrounds) and
                the plain source-sampling seed.
            n_images, sampling, target_size, cull_radius, scene_idx: forwarded to
                :meth:`encode` (defaults mirror ``configs/infer.yaml``).
            **kwargs: per-call overrides for the engine function (take precedence
                over ``config_block``).
        """
        set_global_seeds(int(seed))
        scene = self.encode(
            images,
            scene_idx=scene_idx,
            target_size=target_size,
            cull_radius=cull_radius,
            n_images=n_images,
            image_sampling=sampling,
            image_seed=int(seed),
        )
        # Plain uses its own seeded generator; thread the seed through so a
        # bare Surflo.reconstruct(..., mode="plain") is reproducible.
        if mode == "plain" and "seed" not in kwargs:
            kwargs["seed"] = int(seed)
        return scene.reconstruct(
            mode=mode, config_block=config_block, expert_cfg=expert_cfg,
            opacity_threshold=opacity_threshold, **kwargs,
        )


class SceneState:
    """Preprocessed single-scene handle: cached global state + interaction methods.

    Built by :meth:`Surflo.encode`. Exposes the preprocessed VGGT data as
    read-only properties and provides:

      * :meth:`velocity`               -- raw flow velocity ``v(x, t)`` (Point 3);
      * :meth:`reconstruct`            -- plain / guided ODE solve (Point 4);
      * :meth:`sample_source`,
        :meth:`lift_to_flow_space`,
        :meth:`unlift_from_flow_space` -- flow-space helpers.

    The scene setup (per-scene normalisation stats + token compression) is
    duplicated from :class:`surflo.inference.plain._PlainRun` and must stay in
    sync with it, so :meth:`velocity` returns exactly what the plain / guided
    ODE loops query per step.
    """

    def __init__(self, model: torch.nn.Module, batch: dict, *, scene_idx: int = 0):
        self.model = model
        self.batch = batch
        self.scene_idx = int(scene_idx)
        self.device = model.device

        # -- Slice single-scene VGGT data (same slicing as _PlainRun) -----------
        self.scene_tokens = [
            t[self.scene_idx:self.scene_idx + 1] if t is not None else None
            for t in batch["aggregated_tokens_list"]
        ]
        vggt_wp = batch.get("vggt_world_points")
        self.scene_wp = (
            vggt_wp[self.scene_idx:self.scene_idx + 1] if vggt_wp is not None else None
        )
        self.patch_start_idx = batch["patch_start_idx"]
        self.cull_radius = (
            batch["cull_radius"][self.scene_idx] if "cull_radius" in batch else None
        )

        # -- Per-scene normalisation stats (same as _PlainRun) ------------------
        self.scene_mean = self.scene_std = None
        self.cull_mean = self.cull_std = None
        if model.per_scene_normalize and self.scene_wp is not None:
            self.scene_mean, self.scene_std = model._compute_scene_stats(self.scene_wp)
            if model.renormalize_after_cull and self.cull_radius is not None:
                self.cull_mean, self.cull_std = self.scene_mean, self.scene_std
                self.scene_mean, self.scene_std = model._compute_post_cull_scene_stats(
                    self.scene_wp, self.cull_mean, self.cull_std, self.cull_radius,
                )

        # -- Compressed tokens: the scene's global/latent state (once) ----------
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            self.compressed_tokens, _ = model.surface_net.get_compressed_tokens(
                self.scene_tokens, self.patch_start_idx,
                vggt_world_points=self.scene_wp,
                scene_mean=self.scene_mean, scene_std=self.scene_std,
                cull_radius=self.cull_radius,
                cull_mean=self.cull_mean, cull_std=self.cull_std,
            )
            self.compressed_camera_tokens = None
            if (
                model.surface_net.use_camera_tokens
                and model.surface_net.encode_camera_tokens_separately
            ):
                self.compressed_camera_tokens = (
                    model.surface_net.get_compressed_camera_tokens(
                        self.scene_tokens, self.patch_start_idx,
                    )
                )

        # -- Lazily-computed DepthAnything-3 guidance priors (cached) -----------
        self._monodepths: Optional[torch.Tensor] = None
        self._normal_guidances: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Preprocessed-data accessors (Point 2 outputs)
    # ------------------------------------------------------------------
    @property
    def estimate_normals(self) -> bool:
        return self.model.estimate_normals

    @property
    def images(self) -> torch.Tensor:
        """Resized RGB input images for this scene, ``(N, 3, H, W)`` in ``[0, 1]``."""
        return self.batch["rgb_images"][self.scene_idx]

    @property
    def intrinsics(self) -> torch.Tensor:
        """VGGT intrinsics, ``(N, 3, 3)``."""
        return self.batch["vggt_intrinsics"][self.scene_idx]

    @property
    def extrinsics(self) -> torch.Tensor:
        """VGGT extrinsics (world-to-camera), ``(N, 3, 4)``."""
        return self.batch["vggt_extrinsics"][self.scene_idx]

    @property
    def depth(self) -> torch.Tensor:
        """VGGT depth maps, ``(N, H, W, 1)``."""
        return self.batch["vggt_depth"][self.scene_idx]

    @property
    def confidence(self) -> Optional[torch.Tensor]:
        """VGGT depth-confidence maps, ``(N, H, W)`` (``None`` if unavailable)."""
        conf = self.batch.get("vggt_depth_conf")
        return conf[self.scene_idx] if conf is not None else None

    @property
    def world_points(self) -> torch.Tensor:
        """VGGT back-projected world points, ``(N, H, W, 3)``."""
        return self.batch["vggt_world_points"][self.scene_idx]

    @property
    def cameras(self):
        """The VGGT :class:`MultiCameras` for the encoded batch.

        ``encode`` produces a single scene (``B = 1``), so this container holds
        exactly this scene's cameras. Use :attr:`intrinsics` / :attr:`extrinsics`
        for per-view tensors.
        """
        return self.batch["vggt_cameras"]

    @property
    def global_state(self) -> torch.Tensor:
        """Compressed global latent tokens (the scene's global state), ``(1, K, D)``."""
        return self.compressed_tokens

    # ------------------------------------------------------------------
    # Point 3 -- flow velocity
    # ------------------------------------------------------------------
    @torch.no_grad()
    def velocity(self, x: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Raw flow-matching velocity ``v(x, t)`` in flow space.

        Args:
            x: flow-space points ``(P, D)`` (e.g. from :meth:`sample_source`).
            t: scalar time in ``[0, 1]``.

        Returns:
            Flow-space velocity ``(P, D)`` (float32).

        This is the exact body of ``_PlainRun.flow_velocity`` -- the same value
        the plain / guided ODE loops query per step -- reusing the cached tokens
        and stats, so it stays consistent with :meth:`reconstruct`. Following the
        runner, the model forward uses the raw ``t`` while the (target-mode only)
        target->velocity conversion uses ``t`` clamped to ``<= 1 - 5e-3``.
        """
        model = self.model
        t_raw = torch.as_tensor(float(t), device=self.device)
        t_clamped = torch.clamp_max(t_raw, 1 - 5e-3)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model.surface_net(
                x, t_raw, self.scene_tokens, self.patch_start_idx,
                vggt_world_points=self.scene_wp,
                compressed_tokens=self.compressed_tokens,
                compressed_camera_tokens=self.compressed_camera_tokens,
                scene_mean=self.scene_mean, scene_std=self.scene_std,
                cull_radius=self.cull_radius,
                cull_mean=self.cull_mean, cull_std=self.cull_std,
            )
            prediction = result[0] if isinstance(result, tuple) else result
            if model.prediction_mode == "target":
                velocity = model.path.target_to_velocity(
                    x_1=prediction.unsqueeze(0), x_t=x.unsqueeze(0), t=t_clamped,
                ).squeeze(0)
            else:
                velocity = prediction
        return velocity.float()

    @torch.no_grad()
    def predict_target(self, x: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Convenience: flow-space endpoint estimate ``x + (1 - t) * v(x, t)``.

        A single Euler step to ``t = 1`` from ``(x, t)``; handy for a quick
        "where would this point land?" probe. Still flow space -- use
        :meth:`unlift_from_flow_space` for world coordinates.
        """
        t_val = float(t)
        v = self.velocity(x, t_val)
        return x + (1.0 - t_val) * v

    # ------------------------------------------------------------------
    # Flow-space helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_source(self, n_points: int, seed: Optional[int] = None) -> torch.Tensor:
        """Sample a flow-space source cloud ``(n_points, D)`` for this scene.

        Uses the same per-chunk sampling as the plain runner. Pass ``seed``
        for a reproducible draw.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return self.model.sample_from_source_distribution(
            n_points=int(n_points), batch_size=1,
            vggt_world_points=self.scene_wp, cull_radius=self.cull_radius,
            scene_mean=self.scene_mean, scene_std=self.scene_std,
            cull_mean=self.cull_mean, cull_std=self.cull_std,
            generator=generator,
        ).squeeze(0)

    @torch.no_grad()
    def lift_to_flow_space(self, points_world: torch.Tensor) -> torch.Tensor:
        """World-space points ``(P, N)`` (N in {3, 6}) -> flow space ``(P, D)``."""
        return self.model.lift_points_to_flow_space(
            points_world.unsqueeze(0), self.scene_mean, self.scene_std,
        ).squeeze(0)

    @torch.no_grad()
    def unlift_from_flow_space(self, points_flow: torch.Tensor) -> torch.Tensor:
        """Flow-space points ``(P, D)`` -> world space ``(P, N)`` (N in {3, 6})."""
        return self.model.unlift_points_from_flow_space(
            points_flow.unsqueeze(0), self.scene_mean, self.scene_std,
        ).squeeze(0)

    # ------------------------------------------------------------------
    # Monodepth-expert priors (for guided modes)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_guidance_experts(
        self,
        expert_cfg: Optional[Union[dict, DictConfig]] = None,
        *,
        compute_monodepths: bool = True,
        compute_normals: bool = True,
        expert: Optional[Any] = None,
        cache: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Run the DepthAnything-3 monodepth expert for this scene.

        Mirrors ``scripts/infer.py``'s ``_maybe_inject_experts``: runs the
        (cached) DA3 model and returns the guidance priors ready to hand to
        :meth:`reconstruct` (``mode="guided"``)::

            {"monodepths": (N, 1, H, W) or None,
             "normal_guidances": (N, 3, H, W) or None}

        both at the input image resolution and ordered to match
        :attr:`images`. Settings come from ``expert_cfg`` (e.g. a Hydra
        ``cfg.expert`` node, a dict, or ``None`` -> the bundled defaults:
        ``monodepth_id``, ``pred_res``, ``process_res_method``,
        ``conf_threshold_for_normal_guidance``).

        By default results are cached on the scene (``cache=True``), so repeated
        :meth:`reconstruct` calls reuse them. Normals require ``vggt_depth`` /
        ``vggt_depth_conf`` in the batch (present for VGGT-preprocessed scenes).
        """
        settings = _resolve_expert_settings(_expert_cfg_to_dict(expert_cfg))
        want_mono = bool(compute_monodepths)
        want_norm = bool(compute_normals)
        result: Dict[str, Optional[torch.Tensor]] = {
            "monodepths": None, "normal_guidances": None,
        }
        if not (want_mono or want_norm):
            return result

        if cache:
            need_mono = want_mono and self._monodepths is None
            need_norm = want_norm and self._normal_guidances is None
        else:
            need_mono, need_norm = want_mono, want_norm

        if need_mono or need_norm:
            from .inference.expert import compute_guidance_tensors

            exp = expert if expert is not None else _get_monodepth_expert(
                settings["monodepth_id"], self.device,
            )
            mono, norm = compute_guidance_tensors(
                batch=self.batch, scene_idx=self.scene_idx, expert=exp,
                pred_res=int(settings["pred_res"]),
                process_res_method=str(settings["process_res_method"]),
                conf_threshold_for_normal_guidance=float(
                    settings["conf_threshold_for_normal_guidance"]
                ),
                compute_monodepths=need_mono, compute_normals=need_norm,
                cameras=None, device=self.device,
            )
            if cache:
                if need_mono:
                    self._monodepths = mono
                if need_norm:
                    self._normal_guidances = norm
            else:
                result["monodepths"] = mono if want_mono else None
                result["normal_guidances"] = norm if want_norm else None

        if cache:
            if want_mono:
                result["monodepths"] = self._monodepths
            if want_norm:
                result["normal_guidances"] = self._normal_guidances
        return result

    def _inject_experts(
        self, params: Dict[str, Any], expert_cfg: Optional[Union[dict, DictConfig]],
    ) -> Dict[str, Any]:
        """Fill in ``monodepths`` / ``normal_guidances`` when a guided preset
        asks for them."""
        needs_mono = (
            bool(params.get("use_monodepth_guidance", False))
            and params.get("monodepths") is None
        )
        needs_norm = (
            bool(params.get("use_normal_guidance", False))
            and params.get("normal_guidances") is None
        )
        if not (needs_mono or needs_norm):
            return params

        if expert_cfg is None:
            raise RuntimeError(
                "This preset enables monodepth/normal guidance "
                f"(use_monodepth_guidance={params.get('use_monodepth_guidance', False)}, "
                f"use_normal_guidance={params.get('use_normal_guidance', False)}) but no "
                "`expert_cfg` was given and no `monodepths`/`normal_guidances` tensors "
                "were passed. Either pass expert_cfg=cfg.expert (or {} for defaults) to "
                "auto-run the DepthAnything-3 expert, precompute tensors with "
                "scene.compute_guidance_experts(...), or disable the guidance flags."
            )
        cfg = _expert_cfg_to_dict(expert_cfg)
        if not bool(cfg.get("enabled", True)):
            raise RuntimeError(
                "expert_cfg.enabled=false but the preset requests monodepth/normal "
                "guidance. Enable the expert, pass the tensors, or disable the flags."
            )

        experts = self.compute_guidance_experts(
            expert_cfg=cfg, compute_monodepths=needs_mono, compute_normals=needs_norm,
        )
        params = dict(params)
        if needs_mono and experts.get("monodepths") is not None:
            params["monodepths"] = experts["monodepths"]
        if needs_norm and experts.get("normal_guidances") is not None:
            params["normal_guidances"] = experts["normal_guidances"]
        return params

    # ------------------------------------------------------------------
    # Point 4 -- ODE solve
    # ------------------------------------------------------------------
    def reconstruct(
        self,
        mode: str = "plain",
        config_block: Optional[Union[dict, DictConfig]] = None,
        expert_cfg: Optional[Union[dict, DictConfig]] = None,
        opacity_threshold: Optional[float] = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Solve the flow ODE and return the reconstructed cloud.

        Dispatches to the same engine entry points as ``scripts/infer.py``:
        ``mode="plain"`` (unguided) or ``"guided"``.
        Parameters may come from a ``config_block`` and/or ``**kwargs`` (which
        take precedence); keys the chosen function does not accept are dropped,
        exactly like the CLI. ``config_block`` must be the *resolved* preset block
        for this mode (e.g. ``cfg.guided`` from a Hydra-composed
        ``infer.yaml``, or a plain dict) -- a raw ``OmegaConf.load`` of
        ``infer.yaml`` will not resolve the config groups.

        For guided modes whose preset turns on ``use_monodepth_guidance`` /
        ``use_normal_guidance`` (e.g. ``guided=default`` / ``minimal`` /
        ``long`` -- every preset except ``no_expert``), pass
        ``expert_cfg`` (a Hydra ``cfg.expert`` node, a dict, or ``{}`` for
        defaults) to auto-run the DepthAnything-3 expert and inject the
        ``monodepths`` / ``normal_guidances`` priors -- exactly what
        ``scripts/infer.py`` does. You may also precompute them with
        :meth:`compute_guidance_experts` and pass the tensors directly as kwargs.

        ``opacity_threshold`` (default ``0.1``) filters only the returned
        *display* cloud: ``points`` / ``normals`` / ``colors`` are set to the
        ``aux_opacities >= threshold`` subset (the equivalent of the CLI's
        ``point_cloud_normals.ply``). The full Gaussian set (``aux_means`` and every
        other ``aux_*``) is **never** culled, so :meth:`extract_mesh` /
        :meth:`color_points` always see all Gaussians -- matching the CLI, whose
        mesh is always built from the full set. Pass ``None`` (or ``<= 0``) to
        keep every point; it is a no-op for ``plain`` mode (no per-point
        opacities).

        Returns the engine result dict: the full Gaussians under ``aux_means`` /
        ``aux_scales`` / ``aux_quats`` / ``aux_opacities`` / ``aux_colors`` /
        ``aux_normals`` / ``aux_colors_sh``, plus the filtered display cloud
        ``points`` / ``normals`` / ``colors``. Feed it to :func:`save_ply`.
        """
        if mode not in _ENGINE_FNS:
            raise ValueError(
                f"Unknown mode {mode!r}; expected one of {sorted(_ENGINE_FNS)}."
            )
        fn = _ENGINE_FNS[mode]

        params: Dict[str, Any] = {}
        if config_block is not None:
            params.update(
                OmegaConf.to_container(config_block, resolve=True)
                if isinstance(config_block, DictConfig)
                else dict(config_block)
            )
        params.update(kwargs)

        if mode == "plain":
            gscale = params.pop("guidance_scale", None)
            if gscale is not None and float(gscale) != 0.0:
                _log.info(
                    f"[api] plain mode is unguided; ignoring guidance_scale={gscale}."
                )
            params = _filter_kwargs_to_signature(fn, params)
            return fn(model=self.model, batch=self.batch, scene_idx=self.scene_idx, **params)

        # Guided modes: build monodepth / normal priors when the preset asks for
        # them (no-op if the flags are off or tensors already supplied).
        params = self._inject_experts(params, expert_cfg)
        params = _filter_kwargs_to_signature(fn, params)
        # Guided variants need fp32 + grads (matches scripts/infer.py).
        with torch.amp.autocast("cuda", enabled=False), torch.enable_grad():
            result = fn(model=self.model, batch=self.batch, scene_idx=self.scene_idx, **params)
        # Keep every Gaussian; expose the opacity-filtered display cloud under
        # points/normals/colors (aux_* stay full for meshing/coloring).
        return _finalize_guided_result(result, opacity_threshold)

    # ------------------------------------------------------------------
    # Meshing (mirrors scripts/infer.py's guided mesh.ply extraction)
    # ------------------------------------------------------------------
    @staticmethod
    def _require_guided_gaussians(result: Dict[str, Any]) -> None:
        """Raise unless ``result`` carries the guided Gaussians meshing needs."""
        means = result.get("aux_means")
        n = int(means.shape[0]) if isinstance(means, torch.Tensor) else 0
        for key in ("aux_scales", "aux_quats", "aux_opacities", "aux_colors"):
            v = result.get(key)
            if not isinstance(v, torch.Tensor) or v.shape[0] != n or n == 0:
                raise ValueError(
                    "extract_mesh needs a guided result (mode='guided'): "
                    "the wrapping mesh is extracted from the full "
                    "guidance Gaussians (aux_means + aux_scales/aux_quats/"
                    "aux_opacities/aux_colors). The given result is missing / has "
                    f"empty '{key}', which is what a plain (unguided) run returns. "
                    "Run reconstruct(mode='guided', ...) first."
                )

    @torch.no_grad()
    def extract_mesh(
        self,
        result: Dict[str, Any],
        *,
        mesh_cfg: Optional[Union[dict, DictConfig]] = None,
        use_refined_cameras: Optional[bool] = None,
        save_path: Optional[Union[str, Path]] = None,
        binary: bool = True,
        **overrides,
    ):
        """Extract a wrapping mesh from a guided reconstruction result.

        The same pipeline as the CLI's ``mesh.ply`` step: it builds the
        optimized Gaussians
        (:func:`~surflo.utils.guided_result.gaussians_from_guided_result`) and the
        *refined* cameras
        (:func:`~surflo.utils.guided_result.build_refined_cameras_from_guided_result`),
        then runs
        :func:`~surflo.extraction.occupancy.wrapping.pivot_extraction_with_binary_search`.

        Camera correction: the guided flow learns a per-view pose correction and
        applies it by transforming the Gaussians (``aux_cam_quats`` /
        ``aux_cam_trans``). For extraction that is inverted into one corrected
        camera per view, exactly like the CLI, so the mesh is built against the
        poses the flow actually rendered against. Controlled by
        ``use_refined_cameras`` (default: the ``mesh_cfg`` value, else ``True``);
        if the result has no ``aux_cam_*`` (e.g. a preset that never learned
        camera pose) the original cameras are used.

        Args:
            result: a guided result dict from :meth:`reconstruct`
                (``mode="guided"``). A plain result raises
                ``ValueError`` -- meshing needs the guidance Gaussians.
            mesh_cfg: extraction settings (a Hydra ``cfg.mesh`` node, a dict, or
                ``None`` for the :data:`_MESH_DEFAULTS`, which mirror
                ``configs/mesh/default.yaml``).
            use_refined_cameras: override the ``mesh_cfg`` camera-correction flag.
            save_path: if given, also write the mesh to this ``.ply`` path.
            binary: binary vs ASCII PLY when ``save_path`` is given.
            **overrides: per-call overrides of any extraction knob (e.g.
                ``n_binary_steps=16``), taking precedence over ``mesh_cfg``.

        Returns:
            The extracted mesh object (``.verts`` / ``.faces`` / ``.verts_colors``).
            Feed it to :func:`save_mesh` (or pass ``save_path``).
        """
        self._require_guided_gaussians(result)

        from .extraction.occupancy.wrapping import pivot_extraction_with_binary_search
        from .utils.guided_result import (
            build_refined_cameras_from_guided_result,
            gaussians_from_guided_result,
        )

        params = dict(_MESH_DEFAULTS)
        if mesh_cfg is not None:
            cfgd = (
                OmegaConf.to_container(mesh_cfg, resolve=True)
                if isinstance(mesh_cfg, DictConfig)
                else dict(mesh_cfg)
            )
            params.update({k: v for k, v in cfgd.items() if k in _MESH_DEFAULTS})
        params.update({k: v for k, v in overrides.items() if k in _MESH_DEFAULTS})

        apply_corr = (
            bool(params["use_refined_cameras"])
            if use_refined_cameras is None
            else bool(use_refined_cameras)
        )

        gs = gaussians_from_guided_result(result, device=self.device)
        cameras = build_refined_cameras_from_guided_result(
            batch=self.batch, scene_idx=self.scene_idx, guided_result=result,
            apply_camera_correction=apply_corr, data_device=str(self.device),
        )
        bg = torch.zeros(3, device=self.device, dtype=torch.float32)
        extract_kwargs = dict(
            sdf_mode="approximate",
            sdf_isosurface_value=float(params["sdf_isosurface_value"]),
            use_regular_pivots=bool(params["use_regular_pivots"]),
            std_factor=float(params["std_factor"]),
            n_pivots=int(params["n_pivots"]),
            use_smallest_axis_as_normal=bool(params["use_smallest_axis_as_normal"]),
            n_points_per_sdf_evaluation=int(params["n_points_per_sdf_evaluation"]),
            use_valid_mask=bool(params["use_valid_mask"]),
            filter_large_edges=bool(params["filter_large_edges"]),
            collapse_large_edges=bool(params["collapse_large_edges"]),
            mtet_on_cpu=bool(params["mtet_on_cpu"]),
            n_binary_steps=int(params["n_binary_steps"]),
            delaunay_method=str(params["delaunay_method"]),
        )
        _log.info(
            f"[api] extracting wrapping mesh (N_gauss={int(gs.means.shape[0])}, "
            f"N_views={len(cameras)}, refined_cameras={apply_corr})."
        )
        mesh = pivot_extraction_with_binary_search(
            views=cameras, gaussians=gs, background=bg,
            kernel_size=float(params["kernel_size"]), **extract_kwargs,
        )
        if save_path is not None:
            save_mesh(mesh, save_path, binary=binary)
        return mesh

    # ------------------------------------------------------------------
    # TSDF-based coloring (mirrors scripts/infer.py's RGB fusion)
    # ------------------------------------------------------------------
    def _scene_rgb(self) -> torch.Tensor:
        """Scene RGB images ``(N, 3, H, W)`` on device: prefer ``rgb_images``,
        fall back to ``images``."""
        imgs = self.batch.get("rgb_images")
        if imgs is None:
            imgs = self.batch.get("images")
        if imgs is None:
            raise RuntimeError("no 'rgb_images'/'images' in the scene batch.")
        return imgs[self.scene_idx].to(self.device)

    @torch.no_grad()
    def color_points(
        self,
        points: torch.Tensor,
        result: Dict[str, Any],
        *,
        use_refined_cameras: bool = True,
        trunc_margin_first_pass: Optional[float] = None,
        trunc_margin_fallback_factor: float = 1.0,
    ) -> torch.Tensor:
        """TSDF-color a point set from the input RGB using the Gaussians' depth.

        A thin wrapper over
        :func:`surflo.extraction.texture.optimize.evaluate_point_colors_via_gaussian_tsdf`
        -- the exact routine ``scripts/infer.py`` uses for
        ``point_cloud_rgb.ply``. It renders each view's RaDe-GS median depth
        from the guided Gaussians, then two-pass TSDF-fuses the scene RGB onto
        ``points`` (first a tight truncation, then a relaxed pass for the misses).

        ``result`` must be a guided result (it supplies the Gaussians and the
        refined cameras); a plain result raises ``ValueError``. Camera correction
        (``aux_cam_quats`` / ``aux_cam_trans``) is inverted into refined cameras
        just like meshing -- toggle with ``use_refined_cameras``.

        Args:
            points: ``(P, 3)`` points to color (e.g. ``result["points"]``).
            result: the guided result from :meth:`reconstruct`.

        Returns:
            ``(P, 3)`` float colors in ``[0, 1]``. Pass them to :func:`save_ply`
            via ``colors=`` to write a colored cloud.
        """
        self._require_guided_gaussians(result)
        from .extraction.texture.optimize import evaluate_point_colors_via_gaussian_tsdf
        from .utils.guided_result import (
            build_refined_cameras_from_guided_result,
            gaussians_from_guided_result,
        )

        gs = gaussians_from_guided_result(result, device=self.device)
        cameras = build_refined_cameras_from_guided_result(
            batch=self.batch, scene_idx=self.scene_idx, guided_result=result,
            apply_camera_correction=bool(use_refined_cameras),
            data_device=str(self.device),
        )
        pts = points.to(self.device).float().reshape(-1, 3)
        return evaluate_point_colors_via_gaussian_tsdf(
            points=pts, gaussians=gs, cameras=cameras, images=self._scene_rgb(),
            trunc_margin_first_pass=trunc_margin_first_pass,
            trunc_margin_fallback_factor=float(trunc_margin_fallback_factor),
        )

    @torch.no_grad()
    def color_mesh(
        self,
        mesh: Any,
        result: Dict[str, Any],
        *,
        use_refined_cameras: bool = True,
        trunc_margin_first_pass: Optional[float] = None,
        trunc_margin_fallback_factor: float = 1.0,
        use_scalable_renderer: bool = True,
        assign_to_mesh: bool = True,
    ) -> torch.Tensor:
        """TSDF-color mesh vertices from the input RGB using the mesh's own depth.

        A thin wrapper over
        :func:`surflo.extraction.texture.optimize.evaluate_mesh_colors_all_vertices`
        -- the same two-pass, mesh-rasterized-depth TSDF fusion ``scripts/infer.py``
        uses to seed textured-mesh colors (``init_color="tsdf"``). It rasterizes
        the mesh's own depth per view, then TSDF-fuses the scene RGB onto the
        vertices (tight first pass + relaxed fallback for occluded vertices).

        ``result`` supplies the refined cameras and must be a guided result
        (a plain result raises ``ValueError``).

        Args:
            mesh: a mesh from :meth:`extract_mesh` (``.verts`` / ``.faces``).
            result: the guided result from :meth:`reconstruct`.
            assign_to_mesh: also store the colors on ``mesh.verts_colors`` (default)
                so :func:`save_mesh` writes a colored mesh.

        Returns:
            ``(V, 3)`` float colors in ``[0, 1]``.
        """
        self._require_guided_gaussians(result)
        from .extraction.texture.optimize import evaluate_mesh_colors_all_vertices
        from .utils.guided_result import build_refined_cameras_from_guided_result

        cameras = build_refined_cameras_from_guided_result(
            batch=self.batch, scene_idx=self.scene_idx, guided_result=result,
            apply_camera_correction=bool(use_refined_cameras),
            data_device=str(self.device),
        )
        colors = evaluate_mesh_colors_all_vertices(
            mesh, cameras, self._scene_rgb(),
            trunc_margin_first_pass=trunc_margin_first_pass,
            trunc_margin_fallback_factor=float(trunc_margin_fallback_factor),
            use_scalable_renderer=bool(use_scalable_renderer),
        )
        if assign_to_mesh:
            mesh.verts_colors = colors.to(mesh.verts.device).float()
        return colors


def save_ply(
    result_or_points: Union[Dict[str, Any], torch.Tensor],
    path: Union[str, Path],
    *,
    which: str = "points",
    color: Optional[str] = "normals",
    colors: Optional[Union[torch.Tensor, "np.ndarray"]] = None,
    binary: bool = True,
) -> int:
    """Write a reconstruction to a PLY point cloud.

    Args:
        result_or_points: a dict returned by :meth:`SceneState.reconstruct`, or a
            raw ``(P, 3)`` points tensor.
        path: output ``.ply`` path (parent dirs are created).
        which: for a result dict, ``"points"`` (the reconstructed surface, default)
            or ``"source"`` (the world-space ``source_points`` cloud, present when
            ``reconstruct(mode="plain", return_source=True)`` was used).
        color: ``"normals"`` to color by the ``(1 - n) / 2`` normal map (default,
            requires normals), or ``None`` for no color. Ignored when ``colors``
            is given.
        colors: explicit ``(P, 3)`` per-point colors, overriding ``color`` -- e.g.
            the output of :meth:`SceneState.color_points`. Accepts float
            ``[0, 1]`` (converted to uint8) or ``uint8`` tensors / arrays.
        binary: write binary PLY (default) vs ASCII.

    Returns:
        Number of points written.
    """
    if isinstance(result_or_points, dict):
        if which == "source":
            points = result_or_points.get("source_points")
            normals = result_or_points.get("source_normals")
            if points is None:
                raise KeyError(
                    "result has no 'source_points'; call "
                    "reconstruct(mode='plain', return_source=True) first."
                )
        elif which == "points":
            points = result_or_points.get("points")
            normals = result_or_points.get("normals")
            if points is None:
                raise KeyError("result dict has no 'points'.")
        else:
            raise ValueError(f"which must be 'points' or 'source'; got {which!r}.")
    else:
        points = result_or_points
        normals = None

    colors_uint8 = None
    if colors is not None:
        colors_uint8 = _to_uint8_colors(colors)
    elif color == "normals" and normals is not None:
        colors_uint8 = normals_to_rgb_uint8(normals)
    return write_points_ply(
        str(path), points, colors_uint8=colors_uint8, normals=normals, binary=binary,
    )


def _to_uint8_colors(colors: Union[torch.Tensor, "np.ndarray"]) -> "np.ndarray":
    """Normalise a ``(P, 3)`` color array to uint8 (float ``[0, 1]`` -> uint8)."""
    if isinstance(colors, torch.Tensor):
        if colors.is_floating_point():
            return uint8_colors_from_floats(colors.detach().cpu())
        return colors.detach().cpu().numpy().astype(np.uint8)
    arr = np.asarray(colors)
    if np.issubdtype(arr.dtype, np.floating):
        return uint8_colors_from_floats(torch.as_tensor(arr))
    return arr.astype(np.uint8)


def save_mesh(
    mesh: Any,
    path: Union[str, Path],
    *,
    binary: bool = True,
) -> Tuple[int, int]:
    """Write an extracted mesh to a PLY file.

    Args:
        mesh: the object returned by :meth:`SceneState.extract_mesh` (``.verts`` /
            ``.faces`` / optional ``.verts_colors``).
        path: output ``.ply`` path (parent dirs are created).
        binary: write binary PLY (default) vs ASCII.

    Returns:
        ``(n_verts, n_faces)`` written. Vertex colors are taken from
        ``mesh.verts_colors`` when present.
    """
    verts_colors = getattr(mesh, "verts_colors", None)
    colors_uint8 = None
    if verts_colors is not None:
        colors_uint8 = (
            (verts_colors.detach().cpu().float().clamp(0, 1) * 255 + 0.5)
            .to(torch.uint8).numpy()
        )
    return write_mesh_ply(
        str(path), mesh.verts, mesh.faces,
        vertex_colors_uint8=colors_uint8, binary=binary,
    )


def load_preset(group: str, name: str = "default") -> Dict[str, Any]:
    """Resolve ``configs/<group>/<name>.yaml`` into a plain dict, ready to pass
    as ``config_block`` / ``expert_cfg`` / ``mesh_cfg`` to the API.

    A lightweight, Hydra-free reader of the config presets: it applies the
    ``defaults: [default, _self_]`` inheritance every non-``default`` preset uses
    (load ``default.yaml`` first, then overlay the named preset's own keys), so
    the result matches what the CLI composes -- without importing Hydra, so it is
    safe to call from inside a Hydra app. It does **not** implement variable
    interpolation or nested defaults; the shipped presets use neither.

    Example::

        from surflo import Surflo, load_preset
        surflo = Surflo.from_checkpoint("surflo_v0.pt")
        result = surflo.reconstruct(
            "images/", mode="guided",
            config_block=load_preset("guided", "minimal"),
            expert_cfg=load_preset("expert"),   # minimal enables the DA3 experts
        )

    Args:
        group: config group directory under ``configs/`` (``guided``, ``plain``,
            ``mesh``, ``expert``, ``texture``, ...).
        name: preset stem (without ``.yaml``). Defaults to ``"default"``.

    Raises:
        FileNotFoundError: if the ``configs/`` tree cannot be found next to the
            package (i.e. running from an installed wheel rather than a source
            checkout) or the requested preset does not exist. The message lists
            the presets that *are* available when the group is found.
    """
    import yaml

    # configs/ lives at the repo root, a sibling of the surflo package -- not
    # inside it (it is not packaged data). Anchor off this file: api.py is at
    # <root>/surflo/api.py, so parents[1] is <root>.
    configs_root = Path(__file__).resolve().parents[1] / "configs"
    group_dir = configs_root / group
    if not group_dir.is_dir():
        raise FileNotFoundError(
            f"Config group {group!r} not found at {group_dir}. `load_preset` "
            f"reads the repo's `configs/` tree, which ships with a source "
            f"checkout but not with an installed wheel. Use a source checkout, "
            f"or build the block yourself and pass it as `config_block=`."
        )

    def _read(stem: str) -> dict:
        f = group_dir / f"{stem}.yaml"
        if not f.is_file():
            available = sorted(p.stem for p in group_dir.glob("*.yaml"))
            raise FileNotFoundError(
                f"Preset {stem!r} not found in {group_dir}. Available: {available}."
            )
        data = yaml.safe_load(f.read_text()) or {}
        data.pop("defaults", None)   # inheritance is resolved here, not by Hydra
        return data

    merged = _read("default") if name != "default" else {}
    merged.update(_read(name))
    return merged


__all__ = [
    "Surflo", "SceneState", "save_ply", "save_mesh", "set_global_seeds", "load_preset",
]
