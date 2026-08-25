"""Surflo quantitative evaluation (dump + metrics in a single pass).

Evaluates a checkpoint on a folder of VGGT-preprocessed scenes (produced by
``scripts/preprocess.py``): for each scene it runs the selected ``predictor``,
aligns the prediction to the GT surface cloud (Umeyama on camera centers +
robust scale-locked ICP), and computes symmetric Chamfer (mean/median/trimmed)
and F1@tau. Per-scene records + an aggregate block are written to a JSON file.

The prediction stays in its own (VGGT / DA3) frame and the GT stays in the
COLMAP frame; Umeyama maps the former onto the latter. The metric core lives in
``surflo.metrics.eval_alignment``.

``predictor`` selects what produces the prediction:
  * ``surflo`` (default) -- the flow-matching model (``mode=plain|guided``);
  * ``vggt``             -- the VGGT-1B backbone vendored in the Surflo model;
  * ``da3``              -- DepthAnything3 (loaded from HuggingFace).
The vggt/da3 baselines are eval-only (see ``surflo.eval``). With
``use_tsdf=true`` they fuse per-view depth into a
multi-resolution TSDF mesh and sample its surface instead of subsampling the
raw point map. Baselines re-run their forward pass per scene, so the caches
must include RGB (``scripts/preprocess.py --save_rgb_images``). ``mode=guided``
presets that enable monodepth / normal guidance (e.g. ``minimal``) likewise run
the DepthAnything-3 expert per scene and need those RGB caches.

Examples (run from the repository root):

    # Surflo, plain (no guidance):
    python scripts/evaluate.py benchmarks=tnt \
        ckpt=/path/to/surflo_v0.pt \
        data_dir=/path/to/TNT-preprocessed \
        output_json=eval_results/tnt_plain.json

    # Surflo, guided (per-scene gsplat refinement). `guided=no_densification`
    # and `cull_opacity_threshold=0.1` are the config defaults; pass
    # num_query_points = 2 * num_dump_points so the opacity cull still leaves
    # the target count (`num_dump_points` defaults to 99999). If you raise
    # `num_dump_points`, raise `chamfer_n_points` with it -- they size the
    # prediction and the GT cloud, and Chamfer is only fair between equal
    # sizes.
    python scripts/evaluate.py benchmarks=dl3dv mode=guided \
        ckpt=/path/to/surflo_v0.pt \
        data_dir=/path/to/DL3DV-val-preprocessed \
        num_query_points=200000 \
        output_json=eval_results/dl3dv_guided.json

    # VGGT baseline (raw point map):
    python scripts/evaluate.py benchmarks=tnt predictor=vggt \
        data_dir=/path/to/TNT-preprocessed \
        output_json=eval_results/tnt_vggt.json

    # VGGT + TSDF mesh:
    python scripts/evaluate.py benchmarks=tnt predictor=vggt use_tsdf=true \
        data_dir=/path/to/TNT-preprocessed \
        output_json=eval_results/tnt_vggt_tsdf.json

    # DepthAnything3 baseline (raw / +TSDF):
    python scripts/evaluate.py benchmarks=tnt predictor=da3 \
        data_dir=/path/to/TNT-preprocessed \
        output_json=eval_results/tnt_da3.json
    python scripts/evaluate.py benchmarks=tnt predictor=da3 use_tsdf=true \
        data_dir=/path/to/TNT-preprocessed \
        output_json=eval_results/tnt_da3_tsdf.json
"""
from __future__ import annotations

import gc
import inspect
import json
import logging
import math
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from surflo.model.loader import load_model
from surflo.data.preprocessed import PreprocessedSceneDataset
from surflo.inference.engine import PhaseTimer
from surflo.metrics.eval_alignment import (
    align_pred_to_gt,
    camera_centers_from_extrinsics,
    chamfer_and_fscore,
    voxel_downsample,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Per-scene culling
# ---------------------------------------------------------------------------
def _compute_scene_stats(world_points: torch.Tensor, mode: str):
    pts = world_points.reshape(-1, 3)
    if mode == "median_dist_to_medianpoint":
        center = pts.median(dim=0).values
    else:
        center = pts.mean(dim=0)
    if mode in ("median_dist_to_barycenter", "median_dist_to_medianpoint"):
        scale = (pts - center).norm(dim=-1).median().clamp(min=1e-6).expand(3)
    elif mode == "std":
        scale = pts.std(dim=0).clamp(min=1e-6)
    else:
        raise ValueError(f"Unknown scene_normalize_mode={mode!r}.")
    return center, scale


def _cull_mask(points, center, scale, cull_radius: float):
    return ((points - center) / scale).norm(dim=-1) < cull_radius


def _resolve_cull_stats(
    vggt_world_points, *, per_scene_normalize, scene_normalize_mode,
    global_spatial_mean, global_spatial_std, device,
):
    if per_scene_normalize:
        return _compute_scene_stats(vggt_world_points, scene_normalize_mode)
    if global_spatial_mean is not None and global_spatial_std is not None:
        return (
            global_spatial_mean.to(device=device, dtype=vggt_world_points.dtype),
            global_spatial_std.to(device=device, dtype=vggt_world_points.dtype),
        )
    return None, None


# ---------------------------------------------------------------------------
# Guided extrinsics correction
# ---------------------------------------------------------------------------
def _apply_cam_pose_correction_to_extrinsics(extrinsics, cam_quats, cam_trans):
    """Compose per-camera learned (R_i, t_i) with VGGT extrinsics -> corrected."""
    from surflo.inference.gaussians import quaternion_rotate_vectors

    N = extrinsics.shape[0]
    R_e = extrinsics[:, :3, :3]
    t_e = extrinsics[:, :3, 3]
    q = torch.nn.functional.normalize(cam_quats, dim=-1)
    eye = torch.eye(3, device=extrinsics.device, dtype=extrinsics.dtype).expand(N, 3, 3)
    cols = [quaternion_rotate_vectors(q, eye[:, :, j]) for j in range(3)]
    R_i = torch.stack(cols, dim=-1)
    R_corr = torch.bmm(R_e, R_i)
    t_corr = torch.bmm(R_e, cam_trans.unsqueeze(-1)).squeeze(-1) + t_e
    out = torch.zeros_like(extrinsics)
    out[:, :3, :3] = R_corr
    out[:, :3, 3] = t_corr
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _merge_timings(*sources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten several ``PhaseTimer.as_dict()`` payloads into one.

    Phase names are disjoint across the sources we combine (eval-side
    ``encode`` / ``vggt_depth_recovery`` vs inference-side ``ode`` /
    ``guidance`` / ``polish``), but summing rather than overwriting keeps the
    result correct if that ever stops being true.
    """
    seconds: Dict[str, float] = {}
    calls: Dict[str, int] = {}
    for src in sources:
        if not src:
            continue
        for k, v in (src.get("seconds") or {}).items():
            seconds[k] = seconds.get(k, 0.0) + float(v)
        for k, v in (src.get("calls") or {}).items():
            calls[k] = calls.get(k, 0) + int(v)
    return {"seconds": seconds, "calls": calls}


def _infer_plain(model, batch, *, cfg, num_query_points, generator, profile=False):
    kwargs: Dict[str, Any] = dict(
        num_steps=int(cfg.num_steps),
        num_query_points=int(num_query_points),
        num_points_per_batch=int(cfg.num_points_per_batch),
        cull_radius=batch["cull_radius"][0] if "cull_radius" in batch else None,
        guidance_scale=float(cfg.guidance_scale),
        generator=generator,
        aggregated_tokens_list=[
            t[0:1] if t is not None else None for t in batch["aggregated_tokens_list"]
        ],
        patch_start_idx=batch["patch_start_idx"],
        world_points=(
            batch["vggt_world_points"][0:1] if batch.get("vggt_world_points") is not None else None
        ),
    )
    # The plain eval path goes through `model.batched_inference`, not
    # `plain_inference` / `_PlainRun`, so the runner's own profiler never fires
    # here. There is no guidance in plain mode, so the whole call *is* the ODE.
    prof = PhaseTimer(enabled=bool(profile), device=model.device)
    with prof.phase("ode"):
        with torch.amp.autocast("cuda", enabled=True, dtype=model.dtype):
            out = model.batched_inference(**kwargs)
    if model.estimate_normals:
        pts, nrm = out
    else:
        pts, nrm = out, None
    return {
        "points": pts.float(),
        "normals": nrm.float() if nrm is not None else None,
        "timings": prof.as_dict(),
    }


# Monodepth expert (DepthAnything-3), built once and shared across scenes so
# guided presets that request monodepth / normal guidance (e.g. ``minimal``)
# work under evaluation too.
_MONO_EXPERT_CACHE = None


def _expert_required(guided_kwargs: Dict[str, Any]) -> Tuple[bool, bool]:
    """Return ``(needs_monodepth, needs_normals)`` for a kwargs dict.

    Each is True only when the corresponding ``use_*_guidance`` flag is set AND
    the tensor isn't already provided.
    """
    needs_mono = bool(guided_kwargs.get("use_monodepth_guidance", False))
    needs_norm = bool(guided_kwargs.get("use_normal_guidance", False))
    if needs_mono and guided_kwargs.get("monodepths") is not None:
        needs_mono = False
    if needs_norm and guided_kwargs.get("normal_guidances") is not None:
        needs_norm = False
    return needs_mono, needs_norm


def _maybe_inject_experts(
    *,
    batch: dict,
    scene_idx: int,
    guided_kwargs: Dict[str, Any],
    expert_cfg: Optional[DictConfig],
    device: torch.device,
) -> Dict[str, Any]:
    """Build monodepth-expert tensors when the preset requests them.

    Runs DA3 per scene as ``scripts/infer.py`` does: derives per-view
    monodepth / normals (VGGT-depth aligned, confidence-gated) and puts
    ``monodepths`` / ``normal_guidances`` into a *copy* of ``guided_kwargs``.
    No-op unless the preset turns the guidance flags on.
    """
    global _MONO_EXPERT_CACHE

    needs_mono, needs_norm = _expert_required(guided_kwargs)
    if not (needs_mono or needs_norm):
        return guided_kwargs

    if expert_cfg is None:
        raise RuntimeError(
            "The selected guided preset requests monodepth / normal "
            "guidance but no `expert` config is present. Add `- expert: default` "
            "to eval.yaml's defaults (or flip the guidance flags off in the preset)."
        )
    if not bool(expert_cfg.get("enabled", True)):
        raise RuntimeError(
            "The selected preset requests monodepth / normal guidance but "
            "`expert.enabled=false`. Either enable the expert or flip "
            "`use_monodepth_guidance` / `use_normal_guidance` off in the preset."
        )

    # Imported here so plain / no-expert runs never touch DepthAnything3.
    from surflo.inference.expert import MonodepthExpert, compute_guidance_tensors
    from surflo.structures.cameras import get_cameras_from_intrinsics_and_extrinsics

    if _MONO_EXPERT_CACHE is None:
        _log.info(
            f"[eval] loading monodepth expert ({expert_cfg.monodepth_id}) "
            f"on {device}..."
        )
        _MONO_EXPERT_CACHE = MonodepthExpert(
            model_id=str(expert_cfg.monodepth_id), device=device,
        )

    scene_imgs = batch.get("rgb_images")
    if scene_imgs is None:
        raise RuntimeError(
            "Monodepth expert needs `batch['rgb_images']` (the raw RGB pixels), "
            "but it is missing from the batch. Preprocess the scenes with "
            "`scripts/preprocess.py --save_rgb_images`."
        )

    cameras = None
    if needs_norm:
        cameras = get_cameras_from_intrinsics_and_extrinsics(
            intrinsics=batch["vggt_intrinsics"][scene_idx],
            extrinsics=batch["vggt_extrinsics"][scene_idx],
            images=scene_imgs[scene_idx],
            data_device=str(device),
        )

    monodepths, normals = compute_guidance_tensors(
        batch=batch,
        scene_idx=scene_idx,
        expert=_MONO_EXPERT_CACHE,
        pred_res=int(expert_cfg.pred_res),
        process_res_method=str(expert_cfg.process_res_method),
        conf_threshold_for_normal_guidance=float(
            expert_cfg.conf_threshold_for_normal_guidance
        ),
        compute_monodepths=needs_mono,
        compute_normals=needs_norm,
        cameras=cameras,
        device=device,
    )

    out = dict(guided_kwargs)
    if needs_mono and monodepths is not None:
        out["monodepths"] = monodepths
        _log.info(f"[eval] injected monodepths {tuple(monodepths.shape)}.")
    if needs_norm and normals is not None:
        out["normal_guidances"] = normals
        _log.info(f"[eval] injected normal_guidances {tuple(normals.shape)}.")
    return out


def _infer_guided(model, batch, *, guided_kwargs, cull_opacity_threshold):
    from surflo.inference.engine import guided_inference

    valid = set(inspect.signature(guided_inference).parameters.keys())
    unknown = sorted(set(guided_kwargs) - valid)
    if unknown:
        _log.info(f"[eval] dropping {len(unknown)} preset kwarg(s) unused by guided_inference: {unknown}")
        guided_kwargs = {k: v for k, v in guided_kwargs.items() if k in valid}

    with torch.amp.autocast("cuda", enabled=False), torch.enable_grad():
        result = guided_inference(model=model, batch=batch, scene_idx=0, **guided_kwargs)

    out: Dict[str, Any] = {
        "points": result["points"].float(),
        "normals": (result.get("normals").float() if result.get("normals") is not None else None),
        "opacities": result["aux_opacities"].float(),
        "cam_quats": result.get("aux_cam_quats"),
        "cam_trans": result.get("aux_cam_trans"),
        "timings": result.get("timings"),
    }
    if cull_opacity_threshold is not None and float(cull_opacity_threshold) > 0:
        keep = out["opacities"] >= float(cull_opacity_threshold)
        _log.info(f"[eval] opacity cull: keeping {int(keep.sum())}/{int(keep.numel())} Gaussians.")
        out["points"] = out["points"][keep]
        if out["normals"] is not None:
            out["normals"] = out["normals"][keep]
        out["opacities"] = out["opacities"][keep]
    return out


# ---------------------------------------------------------------------------
# Per-scene metric
# ---------------------------------------------------------------------------
def _safe_centers(extrinsics):
    if extrinsics is None:
        return None
    c = camera_centers_from_extrinsics(extrinsics)
    return c if torch.isfinite(c).all() else None


def _score_scene(
    *, pred_points, pred_extrinsics, gt_points, gt_extrinsics, cfg, diag_eps=1e-12,
) -> Dict[str, Any]:
    degenerate_reason: Optional[str] = None
    if pred_points.shape[0] == 0:
        degenerate_reason = "empty_pred"

    pred_centers = _safe_centers(pred_extrinsics)
    gt_centers = _safe_centers(gt_extrinsics)
    if (pred_centers is not None and gt_centers is not None
            and pred_centers.shape != gt_centers.shape):
        pred_centers = gt_centers = None

    icp = cfg.icp
    align = align_pred_to_gt(
        pred_points=pred_points, gt_points=gt_points,
        pred_cam_centers=pred_centers, gt_cam_centers=gt_centers,
        icp_iters=int(icp.iters), icp_trim_frac=float(icp.trim_frac),
        icp_init_max_dist_frac=float(icp.init_max_dist_frac),
        icp_final_max_dist_frac=float(icp.final_max_dist_frac),
        icp_huber=bool(icp.huber), voxel_frac=float(cfg.voxel_frac),
    )
    pred_aligned = align.pred_aligned

    n_pre_bbox = int(pred_aligned.shape[0])
    bbox_used = False
    if bool(cfg.gt_bbox_cull.enabled) and gt_points.numel() > 0:
        gt_min = gt_points.min(dim=0).values
        gt_max = gt_points.max(dim=0).values
        pad_frac = float(cfg.gt_bbox_cull.padding_frac)
        if pad_frac > 0.0:
            pad = (gt_max - gt_min) * pad_frac
            gt_min, gt_max = gt_min - pad, gt_max + pad
        keep = ((pred_aligned >= gt_min[None]).all(-1) & (pred_aligned <= gt_max[None]).all(-1))
        if int(keep.sum()) > 0:
            pred_aligned = pred_aligned[keep]
            bbox_used = True
    n_post_bbox = int(pred_aligned.shape[0])

    diag = (gt_points.max(dim=0).values - gt_points.min(dim=0).values).norm().item()
    voxel_size = float(cfg.voxel_frac) * diag if float(cfg.voxel_frac) > 0 else 0.0
    if voxel_size > 0:
        pred_ds = voxel_downsample(pred_aligned, voxel_size)
        gt_ds = voxel_downsample(gt_points, voxel_size)
    else:
        pred_ds, gt_ds = pred_aligned, gt_points

    m = chamfer_and_fscore(pred_ds, gt_ds, tau=float(cfg.tau_frac) * diag, trim=float(cfg.trim))
    diag_safe = max(diag, diag_eps)
    if degenerate_reason is None and not math.isfinite(m.chamfer_mean):
        degenerate_reason = "nan_metric_downstream"

    return {
        "chamfer_mean": m.chamfer_mean, "chamfer_median": m.chamfer_median,
        "chamfer_trimmed": m.chamfer_trimmed,
        "chamfer_mean_norm": m.chamfer_mean / diag_safe,
        "chamfer_median_norm": m.chamfer_median / diag_safe,
        "chamfer_trimmed_norm": m.chamfer_trimmed / diag_safe,
        "f_score": m.f_score, "precision": m.precision, "recall": m.recall, "tau": m.tau,
        "scene_diag": diag, "umeyama_scale": float(align.s.item()),
        "num_pred_points": int(pred_points.shape[0]), "num_gt_points": int(gt_points.shape[0]),
        "num_pred_after_bbox_cull": n_post_bbox,
        "num_pred_dropped_by_bbox_cull": n_pre_bbox - n_post_bbox, "bbox_cull_used": bbox_used,
        "num_pred_after_voxel": int(pred_ds.shape[0]), "num_gt_after_voxel": int(gt_ds.shape[0]),
        "alignment_used_centers": (pred_centers is not None and gt_centers is not None),
        "degenerate": degenerate_reason is not None, "degenerate_reason": degenerate_reason,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
_METRIC_KEYS = (
    "chamfer_mean", "chamfer_median", "chamfer_trimmed",
    "chamfer_mean_norm", "chamfer_median_norm", "chamfer_trimmed_norm",
    "f_score", "precision", "recall",
)


def _aggregate(per_scene: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"num_scenes": len(per_scene)}
    if not per_scene:
        return out
    n_deg = sum(1 for r in per_scene if r.get("degenerate"))
    out["num_degenerate_scenes"] = n_deg
    if n_deg:
        reasons: Dict[str, int] = {}
        for r in per_scene:
            if r.get("degenerate"):
                reasons[r.get("degenerate_reason") or "unknown"] = (
                    reasons.get(r.get("degenerate_reason") or "unknown", 0) + 1)
        out["degenerate_reasons"] = reasons
    for k in _METRIC_KEYS:
        vals = np.array([r[k] for r in per_scene if k in r], dtype=np.float64)
        if vals.size == 0:
            continue
        n_valid = int(np.isfinite(vals).sum())
        out[f"num_valid_{k}"] = n_valid
        out[f"num_nan_{k}"] = int(vals.size - n_valid)
        if n_valid == 0:
            for stat in ("mean", "median", "std", "min", "max", "p25", "p75"):
                out[f"{stat}_{k}"] = float("nan")
            continue
        out.update({
            f"mean_{k}": float(np.nanmean(vals)), f"median_{k}": float(np.nanmedian(vals)),
            f"std_{k}": float(np.nanstd(vals, ddof=0)), f"min_{k}": float(np.nanmin(vals)),
            f"max_{k}": float(np.nanmax(vals)), f"p25_{k}": float(np.nanquantile(vals, 0.25)),
            f"p75_{k}": float(np.nanquantile(vals, 0.75)),
        })
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _to_device(batch: dict, device) -> dict:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            batch[k] = [x.to(device, non_blocking=True) for x in v]
    return batch


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ``predictor`` selects which reconstruction produces the prediction:
    #   surflo (default) -> the flow model (mode=plain|guided, unchanged);
    #   vggt / da3       -> the feed-forward baselines (see surflo.eval).
    predictor = str(cfg.get("predictor", "surflo"))
    if predictor not in ("surflo", "vggt", "da3"):
        raise ValueError(f"Unknown predictor={predictor!r}; expected surflo/vggt/da3.")
    mode = str(cfg.mode)
    if predictor == "surflo" and mode not in ("plain", "guided"):
        raise ValueError(f"Unknown mode={mode!r}; expected plain/guided.")
    use_tsdf = bool(cfg.get("use_tsdf", False))
    if use_tsdf and predictor == "surflo":
        raise ValueError("use_tsdf=true is only supported for the vggt/da3 baselines.")

    seed = int(cfg.get("seed", 42))
    _set_seeds(seed)
    device = torch.device(str(cfg.get("device", "cuda")))
    if predictor == "surflo" and cfg.get("ckpt") is None:
        raise ValueError("Surflo evaluation requires ckpt=<path/to/checkpoint.pt>.")
    # Baselines don't need a Surflo checkpoint (VGGT weights load from the hub;
    # DA3 weights from HuggingFace), so ckpt may be null there.
    model = load_model(cfg.model, ckpt_path=cfg.get("ckpt"), device=device,
                       use_ema=bool(cfg.get("use_ema", True)))

    # --- Baseline (vggt/da3) + optional TSDF setup (eval-only) --------------
    # Imported lazily so a plain Surflo run never touches surflo.eval or the
    # DA3 package (keeps the core Surflo eval path completely unchanged).
    da3_model = None
    da3_process_res = 504
    da3_process_res_method = "upper_bound_resize"
    tsdf_geom_kw: Dict[str, Any] = {}
    tsdf_sample_kw: Dict[str, Any] = {}
    if predictor != "surflo":
        from surflo.eval.baselines import load_da3_model, run_baseline_predictor
        da3_process_res = int(cfg.get("da3_process_res", 504))
        da3_process_res_method = str(cfg.get("da3_process_res_method", "upper_bound_resize"))
        if predictor == "da3":
            da3_model = load_da3_model(
                str(cfg.get("da3_model_id", "depth-anything/DA3-LARGE")), device,
            )
        if use_tsdf:
            from surflo.eval.tsdf_mesh import (
                cull_mesh_by_radius,
                extract_multires_tsdf_mesh,
                sample_points_on_mesh_camera_aware,
            )
            from surflo.structures.cameras import (
                get_cameras_from_intrinsics_and_extrinsics,
            )
            tcfg = cfg.get("tsdf") or {}
            tsdf_geom_kw = dict(
                n_cube_per_axis=int(tcfg.get("n_cube_per_axis", 100)),
                trunc_margin_factor=float(tcfg.get("trunc_margin_factor", 0.025)),
                radius_scales=tuple(
                    float(x) for x in tcfg.get("radius_scales", [1.0, 3.0, 10.0])
                ),
                initial_sdf_value=float(tcfg.get("initial_sdf_value", -1.0)),
            )
            tsdf_sample_kw = dict(
                frustum_cull_before_sampling=bool(
                    tcfg.get("frustum_cull_before_sampling", True)
                ),
                frustum_cull_znear=(
                    float(tcfg["frustum_cull_znear"])
                    if tcfg.get("frustum_cull_znear") is not None else None
                ),
            )

    per_scene_normalize = bool(cfg.get("per_scene_normalize", True))
    scene_normalize_mode = str(cfg.get("scene_normalize_mode", "median_dist_to_medianpoint"))
    global_mean = global_std = None
    sm = cfg.model.get("spatial_mean") if cfg.get("model") is not None else None
    ss = cfg.model.get("spatial_std") if cfg.get("model") is not None else None
    if sm is not None and ss is not None:
        global_mean = torch.as_tensor(list(sm), dtype=torch.float32, device=device)
        global_std = torch.as_tensor(list(ss), dtype=torch.float32, device=device)

    cull_radius_cfg = cfg.cull_radius
    if OmegaConf.is_config(cull_radius_cfg):
        cull_radius_cfg = OmegaConf.to_container(cull_radius_cfg, resolve=True)

    dataset = PreprocessedSceneDataset(
        data_dir=str(cfg.data_dir),
        chamfer_n_points=int(cfg.chamfer_n_points),
        cull_radius=cull_radius_cfg,
        per_scene_normalize=per_scene_normalize,
        scene_normalize_mode=scene_normalize_mode,
        spatial_mean=list(sm) if sm is not None else None,
        spatial_std=list(ss) if ss is not None else None,
        fixed_seed_surface_points=True, seed=seed, load_normals=bool(model.estimate_normals),
        n_views=(int(cfg.n_views) if cfg.get("n_views") is not None else None),
        surface_data_dir=(str(cfg.surface_data_dir) if cfg.get("surface_data_dir") is not None else None),
        n_max_views=int(cfg.get("n_max_views", 16)),
    )

    limit = cfg.get("limit_scenes")
    n_scenes = len(dataset) if limit is None else min(len(dataset), int(limit))
    # Optional scene allow-list, e.g. `scene_ids=[Ignatius]`. Dataset indices are
    # preserved so per-scene seeding matches a full run.
    want = cfg.get("scene_ids")
    if want:
        want = {str(x) for x in want}
        scene_indices = [i for i in range(len(dataset))
                         if dataset.scenes[i]["scene_id"] in want]
        if not scene_indices:
            raise ValueError(f"scene_ids={sorted(want)} matched no scene in {cfg.data_dir}")
    else:
        scene_indices = list(range(n_scenes))
    n_scenes = len(scene_indices)
    cull_pred_to_radius = bool(cfg.get("cull_pred_to_radius", True))
    num_dump_points = cfg.get("num_dump_points")
    chamfer_seed_offset = int(cfg.get("chamfer_seed_offset", 0))
    # Per-phase wall-clock + peak VRAM into every per-scene record. Costs a CUDA
    # sync at each phase boundary, so reported times are marginally pessimistic.
    profile = bool(cfg.get("profile", False))

    per_scene: List[Dict[str, Any]] = []
    t_start = time.time()
    for loop_i, i in enumerate(scene_indices):
        # One peak-VRAM number per scene; only the runtime is split by phase.
        # Reset before the batch load so the scene's own tensors count towards
        # the peak.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            vram_baseline = int(torch.cuda.memory_allocated(device))
        else:
            vram_baseline = 0
        scene_prof = PhaseTimer(enabled=profile, device=device)

        batch = _to_device(dataset.get_batch(i), device)
        scene_id = batch["seq_name"][0].replace("dl3dv_preprocessed_", "")

        # Snapshot GT (COLMAP frame) BEFORE the model aligns it into VGGT frame.
        gt_points = batch["chamfer_target_3d_points"][0].float().clone()
        gt_extrinsics = batch["extrinsics"][0].float().clone()
        if gt_extrinsics.ndim == 3 and gt_extrinsics.shape[-2:] == (4, 4):
            gt_extrinsics = gt_extrinsics[:, :3, :]

        gen = torch.Generator(device=device)
        gen.manual_seed(chamfer_seed_offset + i)

        # Baselines finalize their own prediction (raw radius cull + subsample,
        # or TSDF mesh sampling), so the generic Surflo post-processing below
        # is skipped for them.
        pred_finalized = False

        if predictor == "surflo":
            with scene_prof.phase("encode"):
                with torch.amp.autocast("cuda", enabled=False):
                    model.preprocess_from_cached(batch)
            if mode == "guided" and (
                bool(cfg.guided.get("use_confidence", False))
                or bool(cfg.guided.get("use_normal_guidance", False))
            ):
                # Confidence masking needs vggt_depth_conf and the DA3 normal
                # expert needs vggt_depth too; the token cache stores neither.
                # Timed separately: this is a full extra VGGT pass, an artifact
                # of the cache, not part of the encoder cost being reported.
                with scene_prof.phase("vggt_depth_recovery"):
                    _ensure_vggt_depth(model, batch)

            vggt_extrinsics = batch["vggt_extrinsics"][0].float()

            # `num_query_points` is shared by both modes and lives at the top
            # level. `null` means "match the GT cloud size", so Chamfer sees
            # equal-size clouds and its density bias stays constant across
            # methods (`num_dump_points` caps the prediction for the same
            # reason). In guided mode these only *seed* the Gaussians --
            # densification/pruning still moves the final count.
            nqp = cfg.get("num_query_points")
            nqp = int(nqp) if nqp is not None else int(gt_points.shape[0])

            if mode == "plain":
                pred = _infer_plain(
                    model, batch, cfg=cfg, num_query_points=nqp, generator=gen,
                    profile=profile,
                )
                pred_extrinsics = vggt_extrinsics
            else:
                guided_kwargs = OmegaConf.to_container(cfg.guided, resolve=True)
                guided_kwargs["num_query_points"] = nqp
                # Top-level key, injected like num_query_points -- without this
                # guided_inference falls back to its own default (False).
                guided_kwargs["profile"] = profile
                guided_kwargs = _maybe_inject_experts(
                    batch=batch, scene_idx=0, guided_kwargs=guided_kwargs,
                    expert_cfg=cfg.get("expert"), device=device,
                )
                pred = _infer_guided(
                    model, batch, guided_kwargs=guided_kwargs,
                    cull_opacity_threshold=cfg.get("cull_opacity_threshold"),
                )
                if pred.get("cam_quats") is not None and pred.get("cam_trans") is not None:
                    pred_extrinsics = _apply_cam_pose_correction_to_extrinsics(
                        vggt_extrinsics, pred["cam_quats"].to(device), pred["cam_trans"].to(device),
                    )
                else:
                    pred_extrinsics = vggt_extrinsics
        else:
            # ---- VGGT / DA3 baseline forward pass ----
            base = run_baseline_predictor(
                predictor, model=model, batch=batch, device=device,
                da3_model=da3_model,
                process_res=da3_process_res, process_res_method=da3_process_res_method,
            )
            pred_extrinsics = base["extrinsics"].float()

            # Per-scene stats drive both the radius cull and the TSDF grid
            # bounds, so they MUST be computed in the same world frame as the
            # points they are applied to.
            stats_wp = batch.get("vggt_world_points")
            stats_wp = stats_wp[0] if stats_wp is not None else None
            if predictor == "da3":
                stats_wp = base["points"]
            b_center = b_scale = None
            if stats_wp is not None:
                b_center, b_scale = _resolve_cull_stats(
                    stats_wp, per_scene_normalize=per_scene_normalize,
                    scene_normalize_mode=scene_normalize_mode,
                    global_spatial_mean=global_mean, global_spatial_std=global_std, device=device,
                )
            b_cull_radius = (
                float(batch["cull_radius"][0].item()) if "cull_radius" in batch else None
            )
            target_n = (
                int(num_dump_points) if num_dump_points is not None
                else int(gt_points.shape[0])
            )

            if use_tsdf:
                if b_center is None or b_scale is None:
                    raise RuntimeError(
                        "use_tsdf=true requires per-scene normalization stats "
                        "(per_scene_normalize or spatial_mean/std) and cached "
                        "vggt_world_points."
                    )
                mesh = extract_multires_tsdf_mesh(
                    extrinsics=base["extrinsics"], intrinsics=base["intrinsics"],
                    depths=base["depth"], images=base["tsdf_images"],
                    scene_mean=b_center, scene_scale=b_scale, **tsdf_geom_kw,
                )
                if b_cull_radius is not None:
                    mesh = cull_mesh_by_radius(mesh, b_center, b_scale, b_cull_radius)
                cams = get_cameras_from_intrinsics_and_extrinsics(
                    intrinsics=base["intrinsics"], extrinsics=base["extrinsics"],
                    images=base["tsdf_images"],
                )
                sampled = sample_points_on_mesh_camera_aware(
                    mesh, target_n, cameras=cams, generator=gen, **tsdf_sample_kw,
                )
                pred = {"points": sampled.float(), "normals": None}
            else:
                # Raw baseline pointmap: radius cull (with the VGGT-frame stats),
                # then uniform subsample.
                pts_flat = base["points"].reshape(-1, 3).float()
                if b_cull_radius is not None and b_center is not None and b_scale is not None:
                    keep = _cull_mask(pts_flat.to(device), b_center, b_scale, b_cull_radius)
                    pts_flat = pts_flat[keep.to(pts_flat.device)]
                if pts_flat.shape[0] > target_n:
                    idx = torch.randperm(
                        pts_flat.shape[0], device=pts_flat.device, generator=gen,
                    )[:target_n]
                    pts_flat = pts_flat[idx]
                pred = {"points": pts_flat, "normals": None}

            pred_finalized = True

        # ---- Post-inference radius cull (points only) ----
        if not pred_finalized and cull_pred_to_radius and "cull_radius" in batch:
            cr = float(batch["cull_radius"][0].item())
            vggt_wp = batch.get("vggt_world_points")
            if vggt_wp is not None:
                center, scale = _resolve_cull_stats(
                    vggt_wp[0], per_scene_normalize=per_scene_normalize,
                    scene_normalize_mode=scene_normalize_mode,
                    global_spatial_mean=global_mean, global_spatial_std=global_std, device=device,
                )
                if center is not None:
                    keep = _cull_mask(pred["points"].to(device), center, scale, cr)
                    pred["points"] = pred["points"][keep.to(pred["points"].device)]

        # ---- Optional subsample to num_dump_points ----
        if not pred_finalized and num_dump_points is not None and pred["points"].shape[0] > int(num_dump_points):
            n = pred["points"].shape[0]
            g = gen if gen.device == pred["points"].device else torch.Generator(device=pred["points"].device).manual_seed(chamfer_seed_offset + i)
            idx = torch.randperm(n, device=pred["points"].device, generator=g)[:int(num_dump_points)]
            pred["points"] = pred["points"][idx]

        rec = _score_scene(
            pred_points=pred["points"].to(device), pred_extrinsics=pred_extrinsics.to(device),
            gt_points=gt_points.to(device), gt_extrinsics=gt_extrinsics.to(device), cfg=cfg,
        )
        rec["scene_id"] = scene_id
        # The view count actually loaded for this scene, read from the cache
        # itself rather than from the requested `n_views`, so the record always
        # reflects what was measured.
        rec["n_views"] = int(batch["frame_num"].reshape(-1)[0].item())
        if profile:
            rec["timings"] = _merge_timings(scene_prof.as_dict(), pred.get("timings"))
            if torch.cuda.is_available():
                # `allocated` is what PyTorch's allocator handed out; `reserved`
                # is what it holds from the driver and is the closer match to
                # what nvidia-smi shows. Report both -- they differ a lot.
                rec["peak_vram_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                rec["peak_vram_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
                rec["vram_baseline_bytes"] = vram_baseline
        per_scene.append(rec)
        _log.info(
            f"[eval] ({loop_i + 1}/{n_scenes}) {scene_id}: "
            f"CD={rec['chamfer_mean']:.5f} CD_norm={rec['chamfer_mean_norm']:.5f} "
            f"F1={rec['f_score']:.4f}"
        )
        if (loop_i + 1) % 10 == 0:
            agg = _aggregate(per_scene)
            _log.info(
                f"[eval] running mean CD_norm={agg.get('mean_chamfer_mean_norm', float('nan')):.5f} "
                f"F1={agg.get('mean_f_score', float('nan')):.4f}"
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    aggregate = _aggregate(per_scene)
    out = {
        "data_dir": str(cfg.data_dir),
        "benchmark": cfg.get("benchmark_name"),
        "predictor": predictor,
        "use_tsdf": use_tsdf,
        "mode": mode,
        "ckpt": str(cfg.ckpt) if cfg.get("ckpt") is not None else None,
        "config": {
            "seed": seed, "num_steps": int(cfg.num_steps),
            "cull_radius": cull_radius_cfg,
            "n_views": cfg.get("n_views"),
            "surface_data_dir": cfg.get("surface_data_dir"),
            "per_scene_normalize": per_scene_normalize, "scene_normalize_mode": scene_normalize_mode,
            "cull_pred_to_radius": cull_pred_to_radius, "num_dump_points": num_dump_points,
            "icp": OmegaConf.to_container(cfg.icp, resolve=True),
            "gt_bbox_cull": OmegaConf.to_container(cfg.gt_bbox_cull, resolve=True),
            "voxel_frac": float(cfg.voxel_frac), "tau_frac": float(cfg.tau_frac), "trim": float(cfg.trim),
            "da3_model_id": (str(cfg.get("da3_model_id")) if predictor == "da3" else None),
            "tsdf": (OmegaConf.to_container(cfg.tsdf, resolve=True)
                     if (use_tsdf and cfg.get("tsdf") is not None) else None),
        },
        "aggregate": aggregate,
        "results": per_scene,
        "elapsed_seconds": time.time() - t_start,
        "host": socket.gethostname(),
    }
    out_path = Path(str(cfg.output_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    if predictor == "surflo":
        run_label = f"surflo/{mode}"
    else:
        run_label = f"{predictor}{'+tsdf' if use_tsdf else ''}"
    print("\n" + "=" * 60)
    print(f"Surflo eval ({run_label}) on {cfg.get('benchmark_name') or cfg.data_dir}")
    print(f"  scenes:            {aggregate.get('num_scenes')}")
    print(f"  mean Chamfer_norm: {aggregate.get('mean_chamfer_mean_norm', float('nan')):.6f}")
    print(f"  mean F1@tau:       {aggregate.get('mean_f_score', float('nan')):.4f}")
    print(f"  -> {out_path}")
    print("=" * 60)


@torch.no_grad()
def _ensure_vggt_depth(model, batch: dict) -> None:
    """Recover ``vggt_depth`` / ``vggt_depth_conf`` by re-running VGGT.

    The token cache stores neither map. Guided confidence masking needs the
    confidence; the DA3 normal expert needs *both*, since it aligns its
    monodepth prediction to VGGT depth gated by that confidence. One VGGT pass
    on the cached RGB recovers them (that RGB is the exact ``[0, 1]`` input the
    tokens were produced from).
    """
    if "vggt_depth" in batch and "vggt_depth_conf" in batch:
        return
    images = batch.get("rgb_images", batch.get("images"))
    if images is None:
        _log.warning(
            "[eval] guided run needs VGGT depth / confidence but no rgb_images are "
            "cached; preprocess with --save_rgb_images."
        )
        return
    extra = model.preprocess_images(images=images[0], cull_radius=batch.get("cull_radius"))
    for key in ("vggt_depth", "vggt_depth_conf"):
        if key in extra:
            batch[key] = extra[key]


if __name__ == "__main__":
    main()
