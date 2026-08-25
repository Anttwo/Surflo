"""Minimal, standalone reader for the VGGT-preprocessed scene format.

``scripts/preprocess.py`` turns each scene (images + COLMAP + GT mesh
samples) into a directory containing:

  * ``sample_*.pt``     -- cached VGGT tokens, VGGT + COLMAP cameras, the
                           COLMAP<->VGGT alignment (``alignment_L`` / ``T``),
                           scene extent, and (optionally) ``vggt_world_points``
                           / ``rgb_images``.
  * ``surface_data.npz`` -- the consolidated GT surface points (and normals),
                           stored as shuffled numbered chunks.

This class scans a ``data_dir`` of such scenes and yields, per scene, a
**batched (B=1)** dict ready for :meth:`surflo.model.ffm.FFM.preprocess_from_cached`
plus the GT chamfer cloud + COLMAP cameras used by the evaluation metrics.

It has no dependency on the training dataloader stack. For reproducible
benchmarks it iterates scenes in a deterministic order (one scene per item) and
seeds the GT-point sampling from the item index.

Multi-view support
------------------
The cache files may encode a view count in their name
(``sample_<idx>_views_<NNN>.pt``); bare ``sample_<idx>.pt`` files are treated
as the "primary" branch (``n_max_views`` views, default 16). Three knobs
control view selection:

  * ``n_views``        -- request a specific view count ``N`` per scene
                          (deterministic). ``None`` picks any available file.
  * ``surface_data_dir`` -- override where ``surface_data.npz`` lives
                          (defaults to the scene dir under ``data_dir``; use
                          this when GT is stored separately from the caches).

``n_views`` is strict: a scene with no cache at the requested count raises
rather than substituting a nearby one, so a reported view count always matches
what was actually loaded. Leaving it ``None`` picks uniformly among *all* of a
scene's caches, which on a directory holding several view counts mixes them
across scenes -- set it explicitly there.
"""
from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

_log = logging.getLogger(__name__)

# Token layers cached by the preprocessor (patch layers + last camera-token
# layer). Must match SurfaceNet.intermediate_layer_idx + the camera layer.
USED_TOKEN_LAYER_INDICES = [4, 11, 17, 23]

# ``sample_<idx>_views_<NNN>.pt`` (captures base name + view count) and the
# bare ``sample_<idx>.pt`` primary-branch pattern.
_VIEW_FILE_RE = re.compile(r"^(sample_\d+)_views_(\d+)\.pt$")
_BASE_FILE_RE = re.compile(r"^(sample_\d+)\.pt$")


# ----------------------------------------------------------------------
# Cache-filename helpers
#
# Shared with the training datasets (``training/data/datasets/``) so the two
# stacks agree on how a filename maps to a view count and to an exclusion key.
# ----------------------------------------------------------------------
def parse_n_views(filename: str, n_max_views: int) -> Optional[int]:
    """Map a cache filename to its view count (bare files -> ``n_max_views``).

    Returns ``None`` for anything that is not a recognised cache name.
    """
    m = _VIEW_FILE_RE.match(filename)
    if m is not None:
        return int(m.group(2))
    if _BASE_FILE_RE.match(filename) is not None:
        return n_max_views
    return None


def base_sample_name(filename: str) -> Optional[str]:
    """``sample_0000_views_008.pt`` / ``sample_0000.pt`` -> ``sample_0000``.

    Exclusion lists are keyed on the bare ``<scene>/sample_<idx>.pt`` form, so
    every per-view-count file of a sample must collapse to the same key.
    """
    m = _VIEW_FILE_RE.match(filename) or _BASE_FILE_RE.match(filename)
    return m.group(1) if m is not None else None


def collect_files_by_n(
    directory: Path,
    files_by_n: Dict[int, List[str]],
    *,
    scene_id: str,
    n_max_views: int,
    exclude_samples: Optional[Sequence[str]] = None,
    views_only: bool = False,
) -> None:
    """Accumulate ``directory``'s sample files into ``files_by_n`` (in place).

    ``views_only`` skips bare ``sample_<idx>.pt`` files, for a secondary
    directory that only contributes per-view-count caches. ``exclude_samples``
    is matched against the *base* name, so excluding ``<scene>/sample_0000.pt``
    also drops every ``sample_0000_views_*.pt``.
    """
    for f in sorted(directory.glob("sample_*.pt")):
        if views_only and _VIEW_FILE_RE.match(f.name) is None:
            continue
        n = parse_n_views(f.name, n_max_views)
        if n is None:
            continue
        if exclude_samples:
            base = base_sample_name(f.name)
            if base is not None and f"{scene_id}/{base}.pt" in exclude_samples:
                continue
        files_by_n.setdefault(n, []).append(str(f))


class EmptyCullRegionError(RuntimeError):
    """No GT surface points fall inside the culling radius for a scene."""


class PreprocessedSceneDataset:
    """Read VGGT-preprocessed scenes from ``data_dir`` for evaluation."""

    def __init__(
        self,
        data_dir: str,
        *,
        chamfer_n_points: int = 99999,
        n_points: int = 8192,
        cull_radius: Union[float, Sequence[float], None] = None,
        per_scene_normalize: bool = True,
        scene_normalize_mode: str = "median_dist_to_medianpoint",
        spatial_mean: Optional[Sequence[float]] = None,
        spatial_std: Optional[Sequence[float]] = None,
        fixed_seed_surface_points: bool = True,
        seed: int = 42,
        load_normals: bool = True,
        scene_whitelist: Optional[Sequence[str]] = None,
        exclude_samples: Optional[Sequence[str]] = None,
        n_views: Optional[int] = None,
        surface_data_dir: Optional[str] = None,
        n_max_views: int = 16,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.surface_data_dir = Path(surface_data_dir) if surface_data_dir else None
        self.n_views = None if n_views is None else int(n_views)
        self.n_max_views = int(n_max_views)
        assert self.n_max_views >= 1, f"n_max_views must be >= 1, got {n_max_views}"
        self.chamfer_n_points = int(chamfer_n_points)
        self.n_points = int(n_points)
        self.per_scene_normalize = bool(per_scene_normalize)
        self.scene_normalize_mode = str(scene_normalize_mode)
        self.fixed_seed_surface_points = bool(fixed_seed_surface_points)
        self.seed = int(seed)
        self.load_normals = bool(load_normals)
        self.scene_whitelist = set(scene_whitelist) if scene_whitelist else None
        self.exclude_samples = set(exclude_samples) if exclude_samples else set()

        if isinstance(cull_radius, (list, tuple)):
            assert len(cull_radius) == 2 and cull_radius[0] <= cull_radius[1]
            self.cull_radius_range: Optional[Tuple[float, float]] = (
                float(cull_radius[0]), float(cull_radius[1]),
            )
            self.cull_radius: Optional[float] = self.cull_radius_range[1]
        else:
            self.cull_radius_range = None
            self.cull_radius = None if cull_radius is None else float(cull_radius)

        if spatial_mean is not None and spatial_std is not None:
            self.spatial_mean: Optional[torch.Tensor] = torch.tensor(
                list(spatial_mean), dtype=torch.float32)
            self.spatial_std: Optional[torch.Tensor] = torch.tensor(
                list(spatial_std), dtype=torch.float32)
        else:
            self.spatial_mean = None
            self.spatial_std = None

        _valid = ("std", "median_dist_to_barycenter", "median_dist_to_medianpoint")
        assert self.scene_normalize_mode in _valid, (
            f"Invalid scene_normalize_mode={self.scene_normalize_mode!r}; "
            f"expected one of {_valid}."
        )

        self.scenes = self._scan_scenes()
        if not self.scenes:
            raise RuntimeError(
                f"No valid preprocessed scenes found in {self.data_dir}. "
                f"Each scene dir needs sample_*.pt and surface_data.npz files."
            )
        _log.info(
            f"[preprocessed] {self.data_dir}: found {len(self.scenes)} scenes"
            f"{'' if self.n_views is None else f' (n_views={self.n_views})'}."
        )

    def _resolve_surface_npz(
        self, scene_id: str, scene_rel: Path, scene_dir: Path,
    ) -> Optional[Path]:
        if self.surface_data_dir is not None:
            for cand in (
                self.surface_data_dir / scene_rel / "surface_data.npz",
                self.surface_data_dir / scene_id / "surface_data.npz",
            ):
                if cand.exists():
                    return cand
            return None
        cand = scene_dir / "surface_data.npz"
        return cand if cand.exists() else None

    # ------------------------------------------------------------------
    # Scene discovery
    # ------------------------------------------------------------------
    def _try_scene(self, scene_id: str, scene_dir: Path, scene_rel: Path) -> Optional[dict]:
        if self.scene_whitelist is not None and scene_id not in self.scene_whitelist:
            return None

        surface_npz = self._resolve_surface_npz(scene_id, scene_rel, scene_dir)
        if surface_npz is None:
            _log.warning(f"[preprocessed] {scene_id}: missing surface_data.npz; skipping.")
            return None

        files_by_n: Dict[int, List[str]] = {}
        collect_files_by_n(
            scene_dir, files_by_n, scene_id=scene_id,
            n_max_views=self.n_max_views, exclude_samples=self.exclude_samples,
        )
        if not files_by_n:
            return None
        return {
            "scene_id": scene_id,
            "files_by_n": files_by_n,
            "surface_npz": str(surface_npz),
            "preprocessed_dir": str(scene_dir),
        }

    def _scan_scenes(self) -> List[dict]:
        root = self.data_dir
        if not root.exists():
            raise RuntimeError(f"Preprocessed directory not found: {root}")
        scenes: List[dict] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            # A directory is a flat scene if it holds sample files or a
            # surface_data.npz; otherwise treat it as a split dir of scenes.
            is_flat = (
                any(True for _ in entry.glob("sample_*.pt"))
                or (entry / "surface_data.npz").exists()
            )
            if is_flat:
                s = self._try_scene(entry.name, entry, entry.relative_to(root))
                if s is not None:
                    scenes.append(s)
            else:
                for sub in sorted(entry.iterdir()):
                    if not sub.is_dir():
                        continue
                    s = self._try_scene(sub.name, sub, sub.relative_to(root))
                    if s is not None:
                        scenes.append(s)
        return scenes

    def __len__(self) -> int:
        return len(self.scenes)

    # ------------------------------------------------------------------
    # Sample-file selection (dispatch on requested view count)
    # ------------------------------------------------------------------
    def _select_sample_file(self, scene_info: dict, item_rng: random.Random) -> str:
        """Pick a cache file for the scene, honouring ``self.n_views``.

        Consumes exactly one ``item_rng`` draw so the downstream RNG stream
        (cull radius, target / chamfer sampling) is unaffected by the choice.
        """
        files_by_n: Dict[int, List[str]] = scene_info["files_by_n"]
        if self.n_views is None:
            all_files = [f for n in sorted(files_by_n) for f in files_by_n[n]]
            return item_rng.choice(all_files)

        files = files_by_n.get(self.n_views)
        if not files:
            raise RuntimeError(
                f"Scene {scene_info['scene_id']} has no preprocessed cache for "
                f"n_views={self.n_views} (available: {sorted(files_by_n)}). "
                f"Substituting a different view count would report the requested "
                f"N while measuring another; preprocess this count, pick an "
                f"available one, or exclude the scene."
            )
        return item_rng.choice(files)

    # ------------------------------------------------------------------
    # Surface-point loading (culled in the aligned VGGT frame)
    # ------------------------------------------------------------------
    def _load_surface_points(
        self,
        scene_info: dict,
        n_points: int,
        *,
        alignment_L: torch.Tensor,
        alignment_T: torch.Tensor,
        scene_mean: Optional[torch.Tensor],
        scene_std: Optional[torch.Tensor],
        np_rng: np.random.Generator,
        effective_cull_radius: Optional[float],
    ):
        npz = np.load(scene_info["surface_npz"])
        n_chunks = int(npz["n_chunks"])
        points_accum: List[torch.Tensor] = []
        normals_accum: List[torch.Tensor] = []

        for ci in np_rng.permutation(n_chunks):
            pts = torch.from_numpy(npz[f"points_{ci:03d}"])
            if effective_cull_radius is not None:
                aligned = pts @ alignment_L + alignment_T
                cull_mean = scene_mean if scene_mean is not None else self.spatial_mean
                cull_std = scene_std if scene_std is not None else self.spatial_std
                std_dist = torch.norm((aligned - cull_mean) / cull_std, dim=-1)
                mask = std_dist < effective_cull_radius
                pts = pts[mask]
                if self.load_normals:
                    nrm = torch.from_numpy(npz[f"normals_{ci:03d}"])[mask]
                    normals_accum.append(nrm)
            elif self.load_normals:
                normals_accum.append(torch.from_numpy(npz[f"normals_{ci:03d}"]))
            points_accum.append(pts)
            if sum(p.shape[0] for p in points_accum) >= n_points:
                break

        surface_points = torch.cat(points_accum, dim=0)
        total = surface_points.shape[0]
        if total == 0:
            raise EmptyCullRegionError(
                f"No GT surface points inside cull radius {effective_cull_radius} "
                f"for scene {scene_info['scene_id']}."
            )
        if total > n_points:
            sampled_idx = torch.from_numpy(np_rng.permutation(total)[:n_points])
        else:
            sampled_idx = torch.from_numpy(np_rng.integers(0, total, size=(n_points,)))
        surface_points = surface_points[sampled_idx]
        if self.load_normals:
            surface_normals = torch.cat(normals_accum, dim=0)[sampled_idx]
            return surface_points, surface_normals
        return surface_points

    # ------------------------------------------------------------------
    # Batch assembly
    # ------------------------------------------------------------------
    def _sample_cull_radius(self, item_rng: random.Random):
        if self.cull_radius_range is not None:
            lo, hi = self.cull_radius_range
            return item_rng.uniform(lo, hi), hi
        if self.cull_radius is not None:
            return self.cull_radius, self.cull_radius
        return None, None

    @staticmethod
    def _compute_scene_stats(vggt_world_points: torch.Tensor, mode: str):
        pts = vggt_world_points.reshape(-1, 3)
        if mode == "median_dist_to_medianpoint":
            center = pts.median(dim=0).values
        else:
            center = pts.mean(dim=0)
        if mode in ("median_dist_to_barycenter", "median_dist_to_medianpoint"):
            scale = (pts - center).norm(dim=-1).median().clamp(min=1e-6).expand(3)
        else:
            scale = pts.std(dim=0).clamp(min=1e-6)
        return center, scale

    def get_batch(self, index: int) -> Dict[str, Any]:
        """Return a batched (B=1) dict for scene ``index``.

        GT points (``chamfer_target_3d_points``) stay in the COLMAP frame; the
        VGGT tokens / world points are in the VGGT frame. The FFM model's
        ``preprocess_from_cached`` aligns ``target_*`` / ``chamfer_target_*``
        into the VGGT frame, so callers that need GT in COLMAP frame (the
        eval metric) must snapshot it before preprocessing.
        """
        scene_info = self.scenes[index % len(self.scenes)]

        # Deterministic per-scene seed (epoch=0, fixed-seed sampling).
        seed = index * 100003
        item_rng = random.Random(seed)
        item_np_rng = np.random.default_rng(seed)

        # Order of RNG consumption mirrors FfmDl3dvPreprocessedDataset.get_data:
        # (1) pick sample file, (2) draw cull radius, (3) target sample,
        # (4) derive chamfer rng, (5) chamfer sample.
        sample_file = self._select_sample_file(scene_info, item_rng)
        cached = torch.load(sample_file, map_location="cpu", weights_only=False)

        alignment_L = cached["alignment_L"].squeeze(0)   # (3, 3)
        alignment_T = cached["alignment_T"].squeeze(0)   # (3,)
        if "vggt_world_points" in cached:
            cached["vggt_world_points"] = cached["vggt_world_points"].float()

        effective_cull_radius, max_cull_radius = self._sample_cull_radius(item_rng)

        scene_mean = scene_std = None
        if self.per_scene_normalize and effective_cull_radius is not None:
            if "vggt_world_points" not in cached:
                raise RuntimeError(
                    f"Scene {scene_info['scene_id']}: per_scene_normalize needs "
                    f"vggt_world_points in the cache (preprocess with "
                    f"--save_vggt_world_points)."
                )
            scene_mean, scene_std = self._compute_scene_stats(
                cached["vggt_world_points"].squeeze(0), self.scene_normalize_mode,
            )

        def _load(n, rng):
            nonlocal effective_cull_radius
            while True:
                try:
                    return self._load_surface_points(
                        scene_info, n, alignment_L=alignment_L, alignment_T=alignment_T,
                        scene_mean=scene_mean, scene_std=scene_std, np_rng=rng,
                        effective_cull_radius=effective_cull_radius,
                    )
                except EmptyCullRegionError:
                    if (effective_cull_radius is None or max_cull_radius is None
                            or effective_cull_radius >= max_cull_radius):
                        raise
                    effective_cull_radius = min(effective_cull_radius * 2.0, max_cull_radius)

        target_data = _load(self.n_points, item_np_rng)
        target_normals = None
        if self.load_normals:
            target_points, target_normals = target_data
        else:
            target_points = target_data

        chamfer_rng = np.random.default_rng(item_np_rng.integers(0, 2 ** 31))
        chamfer_data = _load(self.chamfer_n_points, chamfer_rng)
        chamfer_points = chamfer_data[0] if self.load_normals else chamfer_data

        # ---- Stack used token layers ----
        used = [cached["aggregated_tokens_list"][i].squeeze(0) for i in USED_TOKEN_LAYER_INDICES]
        cached_aggregated_tokens = torch.stack(used, dim=0)  # (n_used, N, S, D)

        scene_center = cached["scene_center"]
        if isinstance(scene_center, torch.Tensor) and scene_center.ndim == 1:
            scene_center = scene_center.unsqueeze(0)  # (1, 3)
        scene_radius = cached["scene_radius"]
        if not isinstance(scene_radius, torch.Tensor):
            scene_radius = torch.tensor(float(scene_radius), dtype=torch.float32)

        def _b(x):
            return x.unsqueeze(0) if isinstance(x, torch.Tensor) else x

        batch: Dict[str, Any] = {
            "seq_name": [f"dl3dv_preprocessed_{scene_info['scene_id']}"],
            "sample_file": [sample_file],
            "ids": cached["ids"],
            "frame_num": torch.tensor(len(cached["ids"]), dtype=torch.int32),
            "extrinsics": _b(cached["colmap_extrinsics"]),
            "intrinsics": _b(cached["colmap_intrinsics"]),
            "target_3d_points": _b(target_points),
            "scene_radius": scene_radius.reshape(1),
            "scene_center": _b(scene_center),
            "cached_aggregated_tokens": _b(cached_aggregated_tokens),
            "vggt_extrinsics": _b(cached["vggt_extrinsics"].squeeze(0)),
            "vggt_intrinsics": _b(cached["vggt_intrinsics"].squeeze(0)),
            "alignment_L": _b(alignment_L),
            "alignment_T": _b(alignment_T),
            "chamfer_target_3d_points": _b(chamfer_points),
        }
        if target_normals is not None:
            batch["target_normals"] = _b(target_normals)
        if effective_cull_radius is not None:
            batch["cull_radius"] = torch.tensor([effective_cull_radius], dtype=torch.float32)
        if "vggt_world_points" in cached:
            batch["vggt_world_points"] = _b(cached["vggt_world_points"].squeeze(0))
        if "rgb_images" in cached:
            batch["rgb_images"] = _b(cached["rgb_images"].squeeze(0))
        return batch
