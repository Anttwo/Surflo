"""Variable-view-count variant of :class:`FfmDl3dvPreprocessedDataset`.

Caches are grouped by view count into ``files_by_n``, and the batch sampler
picks which ``N`` to serve per item via ``img_per_seq``. A single directory
(``DL3DV_DIR``) holds every view count as ``sample_<idx>_views_<NNN>.pt``
alongside ``surface_data.npz`` -- what ``scripts/preprocess.py`` produces.

A bare ``sample_<idx>.pt`` counts as ``n_max_views`` views.

Scene-level filtering (``DEFAULT_EXCLUDE_SCENES``, ``exclude_samples``,
``scene_whitelist``, ``_validate_scene``, the ``num_val_scenes`` split,
retry/blacklist on failed scenes) matches the parent class. ``exclude_samples``
is matched on the *base* name, so excluding ``<scene>/sample_0000.pt`` also
drops every ``<scene>/sample_0000_views_*.pt``.

The per-item batch is shaped exactly like the parent's -- only ``frame_num``
and the view-indexed tensor shapes vary with the requested ``N``.
"""
import logging
import random
from pathlib import Path
from typing import Optional

from surflo.data.preprocessed import collect_files_by_n

from .dl3dv_preprocessed import FfmDl3dvPreprocessedDataset


class FfmDl3dvPreprocessedMultiviewDataset(FfmDl3dvPreprocessedDataset):
    """Variant of :class:`FfmDl3dvPreprocessedDataset` that supports a
    variable number of views per item.

    Args:
        DL3DV_DIR: Preprocessed directory. Holds the scene directories with
            their ``surface_data.npz`` and their ``sample_*.pt`` caches (bare
            or per-view-count).
        n_max_views: View count a bare ``sample_<idx>.pt`` stands for, and
            the upper bound of the sampler's range (default 16).
        **kwargs: Forwarded as-is to :class:`FfmDl3dvPreprocessedDataset`
            (split, n_points, cull_radius, scene_normalize_mode, the
            full curriculum bundle, etc.).
    """

    def __init__(
        self,
        common_conf,
        DL3DV_DIR: str = "",
        n_max_views: int = 16,
        **kwargs,
    ):
        # ``n_max_views`` is owned by the parent and set there before the scan
        # (``super().__init__`` triggers ``_build_scene_list`` -> ``_scan_scenes``,
        # our override below), so it is passed through rather than assigned here.
        super().__init__(
            common_conf=common_conf, DL3DV_DIR=DL3DV_DIR,
            n_max_views=n_max_views, **kwargs,
        )

    # ------------------------------------------------------------------
    # Scene discovery (override): group caches by view count.
    # ------------------------------------------------------------------

    def _scan_scenes(self) -> list[dict]:
        """Filesystem scan over ``DL3DV_DIR``. Called only on rank 0.

        A scene is kept iff all of the following hold:
            1. It is whitelisted (when a whitelist is set) and not in
               ``exclude_scenes``.
            2. At least one cache file survives ``exclude_samples``, in either
               directory.
            3. ``surface_data.npz`` exists in the ``DL3DV_DIR`` scene dir
               (:meth:`_validate_scene`).
        """
        def _resolve_root(path: str) -> Path:
            root = Path(path)
            if not root.is_absolute():
                from hydra.utils import get_original_cwd
                root = Path(get_original_cwd()) / root
            return root

        root = _resolve_root(self.DL3DV_DIR)
        if not root.exists():
            raise RuntimeError(f"Preprocessed directory not found: {root}")

        scenes: list[dict] = []
        # Rejection counters, accumulated across the whole scan.
        n_not_in_whitelist = 0   # scene present but not whitelisted
        n_excluded = 0           # scene id in exclude_scenes
        n_no_files = 0           # no cache survives exclude_samples
        n_no_surface = 0         # surface_data.npz missing
        ex_not_in_whitelist: list[str] = []
        ex_excluded: list[str] = []
        ex_no_files: list[str] = []
        _MAX_EX = 5

        def _try_add_scene(scene_id: str, scene_dir: Path) -> bool:
            """Return True if the scene was added, False if it was filtered."""
            nonlocal n_not_in_whitelist, n_excluded, n_no_files, n_no_surface

            if self.scene_whitelist is not None and scene_id not in self.scene_whitelist:
                n_not_in_whitelist += 1
                if len(ex_not_in_whitelist) < _MAX_EX:
                    ex_not_in_whitelist.append(scene_id)
                return False
            if scene_id in self.exclude_scenes:
                n_excluded += 1
                if len(ex_excluded) < _MAX_EX:
                    ex_excluded.append(scene_id)
                return False

            files_by_n: dict[int, list[str]] = {}
            collect_files_by_n(
                scene_dir, files_by_n, scene_id=scene_id,
                n_max_views=self.n_max_views,
                exclude_samples=self.exclude_samples,
            )
            if not files_by_n:
                n_no_files += 1
                if len(ex_no_files) < _MAX_EX:
                    ex_no_files.append(scene_id)
                return False

            if not self._validate_scene(scene_id, scene_dir):
                n_no_surface += 1
                return False

            scenes.append(
                {
                    "scene_id": scene_id,
                    "sample_files": files_by_n.get(self.n_max_views, []),
                    "files_by_n": files_by_n,
                    "preprocessed_dir": str(scene_dir),
                }
            )
            return True

        # Supports both flat (``DL3DV_DIR/SCENE_HASH/``) and split-organised
        # (``DL3DV_DIR/SPLIT/SCENE_HASH/``) layouts. A directory is a scene when
        # it holds sample files or a ``surface_data.npz``; otherwise it is
        # treated as a split directory of scenes.
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            is_flat = (
                any(True for _ in entry.glob("sample_*.pt"))
                or (entry / "surface_data.npz").exists()
            )
            if is_flat:
                _try_add_scene(entry.name, entry)
            else:
                for sub_entry in sorted(entry.iterdir()):
                    if sub_entry.is_dir():
                        _try_add_scene(sub_entry.name, sub_entry)

        # Whitelist sanity check: surface any whitelisted scenes that ended up
        # rejected for *any* reason. This makes preprocessing gaps visible.
        whitelist_missing_msg = ""
        if self.scene_whitelist is not None:
            missing_wl = sorted(set(self.scene_whitelist) - {s["scene_id"] for s in scenes})
            if missing_wl:
                whitelist_missing_msg = (
                    f" | whitelist gap: {len(missing_wl)}/{len(self.scene_whitelist)} "
                    f"whitelisted scenes were not kept (first 5: {missing_wl[:5]})"
                )

        logging.info(
            f"[FfmDl3dvPreprocessedMultiview] scan complete from {root}: "
            f"kept {len(scenes)} scenes. Rejected: "
            f"{n_not_in_whitelist} not-in-whitelist (e.g. {ex_not_in_whitelist}), "
            f"{n_excluded} excluded-scene (e.g. {ex_excluded}), "
            f"{n_no_files} no-cache-files (e.g. {ex_no_files}), "
            f"{n_no_surface} missing-surface-data."
            f"{whitelist_missing_msg}"
        )
        return scenes

    # ------------------------------------------------------------------
    # Sample-file selection (override): dispatch on requested view count.
    # ------------------------------------------------------------------

    def _select_sample_file(
        self,
        scene_info: dict,
        img_per_seq: Optional[int],
        item_rng: random.Random,
    ) -> str:
        """Pick a ``.pt`` cache for the requested number of views ``N``.

        ``N`` is clamped to ``n_max_views`` (the sampler never exceeds it, but
        a bare call may). Uniform random pick over the caches available at
        that exact ``N``.
        """
        n = int(img_per_seq) if img_per_seq is not None else self.n_max_views
        n = min(n, self.n_max_views)

        files = scene_info["files_by_n"].get(n)
        if not files:
            raise RuntimeError(
                f"Scene {scene_info['scene_id']} has no preprocessed cache for "
                f"N={n} views (available: {sorted(scene_info['files_by_n'])}). "
                f"Preprocess this view count, restrict img_nums, or drop the scene."
            )
        return item_rng.choice(files)
