"""Preprocessed DL3DV dataset (single-resolution, 16-view caches).

Loads pre-computed VGGT tokens, alignment matrices and COLMAP cameras from
``.pt`` cache files, and freshly samples GT surface points from
``surface_data.npz`` on every access. Images are not loaded. GT-point
alignment is deferred to the GPU (:meth:`surflo.model.ffm.FFM.preprocess_from_cached`).

See ``training/README.md`` for the expected on-disk cache format.
"""
import logging
import random
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import ListConfig

from surflo.data.preprocessed import collect_files_by_n

from data.base_dataset import BaseDataset


USED_TOKEN_LAYER_INDICES = [4, 11, 17, 23]


class EmptyCullRegionError(RuntimeError):
    """Raised when no GT surface points fall inside the culling radius.

    Caught inside ``get_data`` so the radius can be progressively enlarged
    (up to ``max_cull_radius``) before giving up and letting the outer
    ``BaseDataset.__getitem__`` retry on a different scene.
    """
    pass


# Samples to exclude from training (scene_id/sample_XXXX.pt).
DEFAULT_EXCLUDE_SAMPLES = {
    # Scenes with wrong number of VGGT tokens (S=225 instead of S=745).
    "023633c21f5b4a633a836e5c8eb8e5464bd133db28c3c2fc7324124eeb031105/sample_0000.pt",
    "023633c21f5b4a633a836e5c8eb8e5464bd133db28c3c2fc7324124eeb031105/sample_0001.pt",
    "22657d48067798e7286d409f0c62cba3667f90a3e71f7ad7ba70ab54ed6f885b/sample_0000.pt",
    "22657d48067798e7286d409f0c62cba3667f90a3e71f7ad7ba70ab54ed6f885b/sample_0001.pt",
    "2cbef5aa4038c2674847e25d77d43a7401554f518ef4ff9f36038c6aac7c1e2b/sample_0000.pt",
    "2cbef5aa4038c2674847e25d77d43a7401554f518ef4ff9f36038c6aac7c1e2b/sample_0001.pt",
    "5b93293a6d926ce06f6efa8dbf5429df90d14077f12456c1c48402d007fe9191/sample_0000.pt",
    "5b93293a6d926ce06f6efa8dbf5429df90d14077f12456c1c48402d007fe9191/sample_0001.pt",
    "5d6481db3d3f5d79fba3092a4a8dcb04ba9cb20ce20f66b0dfbd5ca85edfb734/sample_0000.pt",
    "5d6481db3d3f5d79fba3092a4a8dcb04ba9cb20ce20f66b0dfbd5ca85edfb734/sample_0001.pt",
    "69d1d838165c33f0b35abaababa444c06ab789cf6f7a5aa174e06672ffd0f642/sample_0000.pt",
    "69d1d838165c33f0b35abaababa444c06ab789cf6f7a5aa174e06672ffd0f642/sample_0001.pt",
    "6da29bd9d9b51888ac93d545aeedaf7c72a41b56f09f90ff1da3ea01ee200176/sample_0000.pt",
    "6da29bd9d9b51888ac93d545aeedaf7c72a41b56f09f90ff1da3ea01ee200176/sample_0001.pt",
    "785fc27e01679d03934ac0601acf26c54f474ccc23bb313a111f56ff6d4cbdc3/sample_0000.pt",
    "785fc27e01679d03934ac0601acf26c54f474ccc23bb313a111f56ff6d4cbdc3/sample_0001.pt",
    "88e8c9b9d50305d4c35e9f918527b4b4c2a5f69651fe63678278aa4c0fa122c9/sample_0000.pt",
    "88e8c9b9d50305d4c35e9f918527b4b4c2a5f69651fe63678278aa4c0fa122c9/sample_0001.pt",
    "a81d111bd6ccbf36276db910e7baa2ea1a24cc985d344a37cd3526a6b1b1393a/sample_0000.pt",
    "a81d111bd6ccbf36276db910e7baa2ea1a24cc985d344a37cd3526a6b1b1393a/sample_0001.pt",
    "a8a848e57e3e74a18107d8b0166323ede27e80bd9a7ec83cca17bd77bcaf4588/sample_0000.pt",
    "a8a848e57e3e74a18107d8b0166323ede27e80bd9a7ec83cca17bd77bcaf4588/sample_0001.pt",
    "b3d664a8dd5a393196008261e7815266b719c8fec7f4c1cf25c543c2c748b33f/sample_0000.pt",
    "b3d664a8dd5a393196008261e7815266b719c8fec7f4c1cf25c543c2c748b33f/sample_0001.pt",
    "b917b26a4b0387001b6598ca28d7c21f003eb1285bb82fc11cf297e06128c625/sample_0000.pt",
    "b917b26a4b0387001b6598ca28d7c21f003eb1285bb82fc11cf297e06128c625/sample_0001.pt",
    "cc40802589f94594383488ff89249d7be1b0609a3b9cfc980698bf2854406eff/sample_0000.pt",
    "cc40802589f94594383488ff89249d7be1b0609a3b9cfc980698bf2854406eff/sample_0001.pt",
    "e9b49b6588acea0bee7b8ecd0ff34ea24dcb9a8e84e10a124ee8b8bab0983f11/sample_0000.pt",
    "e9b49b6588acea0bee7b8ecd0ff34ea24dcb9a8e84e10a124ee8b8bab0983f11/sample_0001.pt",
    "eb42750070f9e8d5b5569aaffb1a97626a9dcf4ac588f5092824bf183e0e3fa7/sample_0000.pt",
    "eb42750070f9e8d5b5569aaffb1a97626a9dcf4ac588f5092824bf183e0e3fa7/sample_0001.pt",
    "f65f9db36a9cdabfe710325328f64da9d59792cc6b1930ef9cade883e0afd313/sample_0000.pt",
    "f65f9db36a9cdabfe710325328f64da9d59792cc6b1930ef9cade883e0afd313/sample_0001.pt",
    # Scenes with missing sample files (surface_data exists but no samples).
    "a7cc00c501c536cd0355f8175816e1436873e90213f49a36ba8ffeb1670aeb6b/sample_0000.pt",
    "a7cc00c501c536cd0355f8175816e1436873e90213f49a36ba8ffeb1670aeb6b/sample_0001.pt",
    "d2734f43d611a7f596aede22b8eeaafad8c20698fab9c9b6a40a1300dfd45831/sample_0000.pt",
    "d2734f43d611a7f596aede22b8eeaafad8c20698fab9c9b6a40a1300dfd45831/sample_0001.pt",
    "0ef46c798c1a3198206bf16719e30230837ca610ec1bbb6e68c089b569a92536/sample_0000.pt",
    "0ef46c798c1a3198206bf16719e30230837ca610ec1bbb6e68c089b569a92536/sample_0001.pt",
    "2bf0be73ee7a0f5982ca5885de696ff03888a15489be6628b6fc0f1d51356b1d/sample_0000.pt",
    "2bf0be73ee7a0f5982ca5885de696ff03888a15489be6628b6fc0f1d51356b1d/sample_0001.pt",
    "4d24c475f7a213e8c0c4021979b3f2dc1df7759e946d56824b4d6446e82847ee/sample_0000.pt",
    "4d24c475f7a213e8c0c4021979b3f2dc1df7759e946d56824b4d6446e82847ee/sample_0001.pt",
}

# Scene ids to exclude outright (22, derived from the list above).
#
# Both defects above are properties of the *scene*, not of one sample index:
# the token count follows from the scene's image resolution, and it is
# identical across sample indices. Keying on the scene is therefore what
# actually holds -- a sample-keyed entry only drops the scene as long as every
# sample index it owns happens to be listed, and it matches nothing at all on
# the per-view-count caches (``sample_0000_views_016.pt``) that
# ``scripts/preprocess.py`` produces.
DEFAULT_EXCLUDE_SCENES = {s.split("/")[0] for s in DEFAULT_EXCLUDE_SAMPLES}


class FfmDl3dvPreprocessedDataset(BaseDataset):
    """Dataset that loads pre-computed VGGT tokens, alignment matrices, and
    COLMAP cameras from ``.pt`` files, then freshly samples GT surface points
    from disk on every access.

    Images are **not** loaded — only their paths are stored in the batch so
    they can be loaded lazily when needed (e.g. for visualization).

    The batch contains everything the model needs *except* the alignment of
    GT points, which is performed on-GPU by :meth:`FFM.preprocess_from_cached`.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DL3DV_DIR: str = "",
        n_points: int = 2**13,
        n_max_views: int = 16,
        chamfer_n_points: int | None = None,
        num_val_scenes: int | None = None,
        len_train: int = 100000,
        len_test: int = 10000,
        cull_radius: float | list[float] | None = None,
        spatial_mean: list[float] | None = None,
        spatial_std: list[float] | None = None,
        per_scene_normalize: bool = False,
        scene_normalize_mode: str = "std",  # "std", "median_dist_to_barycenter", or "median_dist_to_medianpoint"
        load_normals: bool = False,
        fixed_seed_surface_points: bool = False,
        exclude_samples: list[str] | None = None,
        exclude_scenes: list[str] | None = None,
        scene_whitelist: str | None = None,
        cull_curriculum_warmup_epochs: int = 0,
        cull_curriculum_law: str = "power",
        cull_curriculum_alpha_init: float = 3.0,
        cull_curriculum_k_init: float = 3.0,
        cull_curriculum_mode: float = 1.0,
        cull_curriculum_max_progress: float = 0.85,
    ):
        super().__init__(common_conf=common_conf)

        self.split = split

        self.DL3DV_DIR = hydra.utils.to_absolute_path(DL3DV_DIR)
        self.n_points = n_points
        # Owned here (not by the multi-view subclass) because ``_scan_scenes``
        # needs it, and that runs from ``_build_scene_list()`` below.
        self.n_max_views = int(n_max_views)
        assert self.n_max_views >= 1, f"n_max_views must be >= 1, got {n_max_views}"
        self.chamfer_n_points = chamfer_n_points
        self.per_scene_normalize = per_scene_normalize

        # cull_radius: float -> fixed for all scenes;
        #              [min, max] -> random per-scene; None -> no culling.
        # NOTE: Hydra/OmegaConf passes list-valued config fields as ListConfig,
        # not Python list, so we must include it in the isinstance check.
        if isinstance(cull_radius, (list, tuple, ListConfig)):
            assert len(cull_radius) == 2 and cull_radius[0] <= cull_radius[1], (
                f"cull_radius interval must be [min, max] with min <= max, got {cull_radius}"
            )
            self.cull_radius_range = (float(cull_radius[0]), float(cull_radius[1]))
            self.cull_radius = self.cull_radius_range[1]  # upper bound for chunk loading
        else:
            self.cull_radius_range = None
            self.cull_radius = cull_radius

        # Curriculum sampling: at epoch 0, draw radii from a mode-pinned
        # distribution on [min, max]; both laws relax to Uniform(min, max)
        # at the end of the warmup so training is bias-free once the
        # model has learned the easy cases.
        #
        # Two laws are supported via ``cull_curriculum_law``:
        #
        # - "beta": Beta(alpha, beta) mapped to [min, max], with the mode
        #   pinned at ``cull_curriculum_mode``. ``alpha`` anneals linearly
        #   from ``cull_curriculum_alpha_init`` down to 1.0 over
        #   ``warmup_epochs``; ``beta`` is derived at each epoch from
        #   ``alpha`` and the fixed mode so the peak stays put while the
        #   density widens. At alpha=1.0 the formula gives beta=1.0, i.e.
        #   Beta(1, 1) = Uniform(min, max).
        #
        # - "power": two-sided power law, peaked at ``cull_curriculum_mode``.
        #   Given exponent k, sample u ~ Uniform(0, 1), v = u**k (biased
        #   toward 0 when k > 1), then pick a side of ``mode`` with
        #   probabilities (mode-min)/(max-min) and (max-mode)/(max-min)
        #   and reflect v toward ``mode``:
        #       left:  r = mode - v * (mode - min)
        #       right: r = mode + v * (max - mode)
        #   k anneals linearly from ``cull_curriculum_k_init`` -> 1.0 over
        #   warmup_epochs; at k=1 the two sides each become Uniform on
        #   their half-interval and the mixture equals Uniform(min, max),
        #   so the curriculum again smoothly relaxes to no-bias sampling.
        #   The mode can sit at an endpoint (``mode == min`` collapses to
        #   the classic ``r = min + u**k * (max - min)``; ``mode == max``
        #   is its mirror).
        #
        # Active only on the train split with an interval cull_radius,
        # warmup_epochs > 0, and a shape parameter > 1.0 for the selected
        # law.
        assert cull_curriculum_warmup_epochs >= 0, (
            f"cull_curriculum_warmup_epochs must be >= 0, got {cull_curriculum_warmup_epochs}"
        )
        assert cull_curriculum_law in ("beta", "power"), (
            f"cull_curriculum_law must be one of ('beta', 'power'), got "
            f"{cull_curriculum_law!r}"
        )
        assert cull_curriculum_alpha_init >= 1.0, (
            f"cull_curriculum_alpha_init must be >= 1.0, got {cull_curriculum_alpha_init}"
        )
        assert cull_curriculum_k_init >= 1.0, (
            f"cull_curriculum_k_init must be >= 1.0, got {cull_curriculum_k_init}"
        )
        assert 0.0 <= cull_curriculum_max_progress <= 1.0, (
            f"cull_curriculum_max_progress must be in [0.0, 1.0], got "
            f"{cull_curriculum_max_progress}"
        )
        self.cull_curriculum_warmup_epochs = int(cull_curriculum_warmup_epochs)
        self.cull_curriculum_law = str(cull_curriculum_law)
        self.cull_curriculum_alpha_init = float(cull_curriculum_alpha_init)
        self.cull_curriculum_k_init = float(cull_curriculum_k_init)
        self.cull_curriculum_mode = float(cull_curriculum_mode)
        self.cull_curriculum_max_progress = float(cull_curriculum_max_progress)
        _shape_init = (
            self.cull_curriculum_alpha_init
            if self.cull_curriculum_law == "beta"
            else self.cull_curriculum_k_init
        )
        self._cull_curriculum_active = (
            split == "train"
            and self.cull_radius_range is not None
            and self.cull_curriculum_warmup_epochs > 0
            and _shape_init > 1.0
        )
        if self._cull_curriculum_active:
            lo, hi = self.cull_radius_range
            if self.cull_curriculum_law == "beta":
                # Beta's mode formula (alpha-1)/(alpha+beta-2) is only
                # defined for an interior mode; endpoints collapse the
                # shape.
                assert lo < self.cull_curriculum_mode < hi, (
                    f"cull_curriculum_mode must lie strictly inside the cull_radius "
                    f"interval ({lo}, {hi}) for law='beta', got {self.cull_curriculum_mode}"
                )
            else:
                # Power law allows mode at either endpoint (degenerate
                # single-sided power law).
                assert lo <= self.cull_curriculum_mode <= hi, (
                    f"cull_curriculum_mode must lie inside the cull_radius "
                    f"interval [{lo}, {hi}] for law='power', got {self.cull_curriculum_mode}"
                )
            logging.info(
                f"[cull-curriculum] Active on split={split}: law="
                f"{self.cull_curriculum_law}, shape_init={_shape_init}, mode="
                f"{self.cull_curriculum_mode}, warmup_epochs="
                f"{self.cull_curriculum_warmup_epochs}, max_progress="
                f"{self.cull_curriculum_max_progress}, range="
                f"{self.cull_radius_range}."
            )
        self.fixed_seed_surface_points = fixed_seed_surface_points
        if self.fixed_seed_surface_points:
            print(f"[INFO] Split={split}: Fixing surface points sampling to be the same for all epochs.")
        _valid_modes = ("std", "median_dist_to_barycenter", "median_dist_to_medianpoint")
        assert scene_normalize_mode in _valid_modes, f"Invalid scene_normalize_mode: {scene_normalize_mode}. Must be one of {_valid_modes}"
        self.scene_normalize_mode = scene_normalize_mode
        self.load_normals = load_normals
        if exclude_samples is not None:
            self.exclude_samples = set(exclude_samples)
        else:
            self.exclude_samples = DEFAULT_EXCLUDE_SAMPLES
        # Kept independent of ``exclude_samples``: deriving scene ids from a
        # deliberately partial sample list would silently over-exclude.
        if exclude_scenes is not None:
            self.exclude_scenes = set(exclude_scenes)
        else:
            self.exclude_scenes = DEFAULT_EXCLUDE_SCENES

        if scene_whitelist is not None:
            wl_path = Path(hydra.utils.to_absolute_path(scene_whitelist))
            self.scene_whitelist: set[str] | None = {
                line.strip() for line in wl_path.read_text().splitlines() if line.strip()
            }
            logging.info(f"Loaded scene whitelist with {len(self.scene_whitelist)} IDs from {wl_path}")
        else:
            self.scene_whitelist = None

        if self.cull_radius is not None and not per_scene_normalize:
            assert spatial_mean is not None and spatial_std is not None, (
                "spatial_mean and spatial_std must be provided when cull_radius is set"
                " (unless per_scene_normalize is enabled)"
            )
        if spatial_mean is not None and spatial_std is not None:
            self.spatial_mean = torch.tensor(spatial_mean, dtype=torch.float32)
            self.spatial_std = torch.tensor(spatial_std, dtype=torch.float32)
        else:
            self.spatial_mean = None
            self.spatial_std = None

        self.inside_random = common_conf.inside_random
        self.training = common_conf.training

        if chamfer_n_points is not None:
            logging.info(f"Will also load {self.chamfer_n_points} GT points for chamfer evaluation")

        self.failed_scenes = set()
        self.max_retries_per_getitem = 10

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        # True only for the val split below, where ``len_train`` is set to the
        # scene count; elsewhere it is a fixed budget and must survive pruning.
        self._len_follows_scene_count = False

        self._build_scene_list()

        # Split scenes into disjoint train/val sets when num_val_scenes is set.
        if num_val_scenes is not None and num_val_scenes > 0:
            total = self.sequence_list_len
            assert num_val_scenes < total, (
                f"num_val_scenes ({num_val_scenes}) must be smaller than "
                f"total valid scenes ({total})"
            )
            split_rng = random.Random(42)
            shuffled_indices = list(range(total))
            split_rng.shuffle(shuffled_indices)
            val_indices = set(shuffled_indices[:num_val_scenes])

            if split == "test":
                self.scenes = [self.scenes[i] for i in sorted(val_indices)]
                self.len_train = len(self.scenes)
                self._len_follows_scene_count = True
                self.inside_random = False
            else:
                self.scenes = [s for i, s in enumerate(self.scenes) if i not in val_indices]

            self.sequence_list = [s["scene_id"] for s in self.scenes]
            self.sequence_list_len = len(self.scenes)

            logging.info(
                f"[Split] {split} split: kept {self.sequence_list_len} scenes "
                f"(num_val_scenes={num_val_scenes}, total={total})"
            )

        status = "Training" if self.training else "Testing"
        logging.info(
            f"{status}: DL3DV-Preprocessed dataset with {self.sequence_list_len} scenes, "
            f"dataset length: {len(self)}. Scenes per view count: "
            f"{ {n: c for n, c in self.view_count_coverage.items() if c} }"
        )

    # ------------------------------------------------------------------
    # View-count availability
    # ------------------------------------------------------------------

    @property
    def view_count_coverage(self) -> dict[int, int]:
        """How many scenes can serve each view count ``N in [1, n_max_views]``.

        Derived from ``self.scenes`` on access rather than cached, because the
        ``num_val_scenes`` split reassigns ``self.scenes`` after the scan.
        """
        return {
            n: sum(1 for s in self.scenes if s.get("files_by_n", {}).get(n))
            for n in range(1, self.n_max_views + 1)
        }

    @property
    def available_view_counts(self) -> set[int]:
        """View counts at least one scene can serve."""
        return {n for n, c in self.view_count_coverage.items() if c}

    @property
    def files_per_view_count(self) -> dict[int, int]:
        """Total number of cache files at each view count, across all scenes.

        Useful when choosing ``image_num_weights``: it shows how many caches
        back each ``N``, which is what a weight is trading off against.
        """
        return {
            n: sum(len(s.get("files_by_n", {}).get(n, [])) for s in self.scenes)
            for n in range(1, self.n_max_views + 1)
        }

    def restrict_to_view_counts(self, required: set[int]) -> tuple[int, int]:
        """Drop scenes that cannot serve every view count in ``required``.

        The sampler draws one ``N`` for a whole batch but fills it from any
        scene, so a scene missing one of the selectable counts would raise the
        first time that ``N`` came up -- mid-epoch, after training had started.
        Pruning up front keeps only scenes that are usable for every ``N`` the
        run can draw.

        Called by the multi-view dataloader, which is the only component that
        knows the selectable set (``img_nums`` minus the zero-weighted counts).
        Returns ``(n_kept, n_dropped)``.
        """
        if not required:
            return self.sequence_list_len, 0

        keep, dropped_missing = [], {}
        for sc in self.scenes:
            have = sc.get("files_by_n", {})
            missing = sorted(n for n in required if not have.get(n))
            if missing:
                for n in missing:
                    dropped_missing[n] = dropped_missing.get(n, 0) + 1
            else:
                keep.append(sc)

        n_dropped = len(self.scenes) - len(keep)
        if n_dropped == 0:
            return self.sequence_list_len, 0

        self.scenes = keep
        self.sequence_list = [s["scene_id"] for s in keep]
        self.sequence_list_len = len(keep)
        if self._len_follows_scene_count:
            self.len_train = len(keep)

        logging.warning(
            f"[{type(self).__name__}] dropped {n_dropped} scene(s) that do not "
            f"cover every selectable view count {sorted(required)}; "
            f"{len(keep)} scene(s) kept. Scenes missing each count: "
            f"{ {n: c for n, c in sorted(dropped_missing.items())} }. "
            f"Preprocess the missing counts to use them, or give those counts "
            f"weight 0 in image_num_weights."
        )
        if not keep:
            raise RuntimeError(
                f"{type(self).__name__}: every scene was dropped -- no scene in "
                f"{self.DL3DV_DIR} covers all of {sorted(required)}."
            )
        return len(keep), n_dropped

    # ------------------------------------------------------------------
    # Scene discovery
    # ------------------------------------------------------------------

    def _build_scene_list(self):
        """Scan ``DL3DV_DIR`` for scenes with ``sample_*.pt`` files and
        consolidated surface data.

        Supports both flat layouts (``DL3DV_DIR/SCENE_HASH/``) and
        split-organised layouts (``DL3DV_DIR/SPLIT/SCENE_HASH/``).

        In multi-node DDP setups the filesystem scan can yield different
        results across nodes (e.g. NFS caching).  To guarantee every rank
        sees the exact same scene list, rank 0 performs the scan and
        broadcasts the result.
        """
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        scenes: list[dict] | None = None

        if rank == 0:
            scenes = self._scan_scenes()

        # Broadcast the scene list from rank 0 to all other ranks.
        if world_size > 1:
            scenes_container = [scenes]
            dist.broadcast_object_list(scenes_container, src=0)
            scenes = scenes_container[0]

        assert scenes is not None
        self.scenes = scenes
        self.sequence_list = [s["scene_id"] for s in scenes]
        self.sequence_list_len = len(scenes)

        if self.sequence_list_len == 0:
            raise RuntimeError(
                f"No valid preprocessed scenes found in {self.DL3DV_DIR}. "
                f"Each scene dir needs sample_*.pt and surface_data.npz files."
            )

    def _scan_scenes(self) -> list[dict]:
        """Filesystem scan — called only on rank 0."""
        preprocessed_root = Path(self.DL3DV_DIR)
        if not preprocessed_root.is_absolute():
            from hydra.utils import get_original_cwd
            preprocessed_root = Path(get_original_cwd()) / preprocessed_root
        if not preprocessed_root.exists():
            raise RuntimeError(f"Preprocessed directory not found: {preprocessed_root}")

        scenes: list[dict] = []

        def _try_add_scene(scene_dir: Path):
            if not any(scene_dir.glob("sample_*.pt")):
                return False
            scene_id = scene_dir.name
            if self.scene_whitelist is not None and scene_id not in self.scene_whitelist:
                return False
            if scene_id in self.exclude_scenes:
                return False
            # Group by view count so that per-view-count caches
            # (``sample_<idx>_views_<NNN>.pt``) are not mistaken for the
            # ``n_max_views`` branch. Bare ``sample_<idx>.pt`` files map to
            # ``n_max_views``, so a legacy directory is unaffected.
            files_by_n: dict[int, list[str]] = {}
            collect_files_by_n(
                scene_dir, files_by_n, scene_id=scene_id,
                n_max_views=self.n_max_views,
                exclude_samples=self.exclude_samples,
            )
            sample_files = files_by_n.get(self.n_max_views, [])
            if not sample_files:
                return False
            if not self._validate_scene(scene_id, scene_dir):
                return False
            scenes.append(
                {
                    "scene_id": scene_id,
                    "sample_files": sample_files,
                    "files_by_n": files_by_n,
                    "preprocessed_dir": str(scene_dir),
                }
            )
            return True

        for entry in sorted(preprocessed_root.iterdir()):
            if not entry.is_dir():
                continue
            if not _try_add_scene(entry):
                # No sample files here — might be a split directory (e.g. 10K/)
                for sub_entry in sorted(entry.iterdir()):
                    if sub_entry.is_dir():
                        _try_add_scene(sub_entry)

        return scenes

    def _validate_scene(self, scene_id: str, preprocessed_dir: Path) -> bool:
        surface_file = preprocessed_dir / "surface_data.npz"
        if not surface_file.exists():
            print(
                f"[WARNING] Missing {surface_file} for scene {scene_id}, skipping. "
                f"Run the VGGT preprocessing step to consolidate surface data."
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Surface point loading
    # ------------------------------------------------------------------

    def _load_surface_points(
        self,
        scene_info: dict,
        n_points: int,
        alignment_L: torch.Tensor,
        alignment_T: torch.Tensor,
        scene_mean: torch.Tensor | None = None,
        scene_std: torch.Tensor | None = None,
        np_rng: np.random.Generator | None = None,
        effective_cull_radius: float | None = None,
    ):
        """Load random chunks from the consolidated ``surface_data.npz``,
        optionally cull using ``spatial_mean``/``spatial_std`` (in the aligned
        VGGT frame), and subsample exactly *n_points*.

        Points are stored in COLMAP frame.  When ``effective_cull_radius`` is
        set the alignment transform is applied temporarily so that the culling
        criterion ``||(x - mean) / std||_2 < effective_cull_radius`` operates in
        the correct coordinate frame.  The returned points are still in COLMAP
        frame (alignment is applied later on GPU by the model).

        Args:
            effective_cull_radius: The cull radius to use for this call.
                When ``cull_radius`` is an interval, the caller samples a
                per-scene value and passes it here.
            np_rng: Optional deterministic RNG. When provided, all random
                choices use this generator instead of the global state,
                ensuring reproducibility across DataLoader workers.
        """
        preproc_dir = Path(scene_info["preprocessed_dir"])
        npz = np.load(str(preproc_dir / "surface_data.npz"))
        n_chunks = int(npz["n_chunks"])

        points_accum = []
        normals_accum = []

        if np_rng is not None:
            chunk_order = np_rng.permutation(n_chunks)
        else:
            chunk_order = np.random.permutation(n_chunks)
        for ci in chunk_order:
            pts = torch.from_numpy(npz[f"points_{ci:03d}"])

            if effective_cull_radius is not None:
                aligned = pts @ alignment_L + alignment_T
                cull_mean = scene_mean if scene_mean is not None else self.spatial_mean
                cull_std = scene_std if scene_std is not None else self.spatial_std
                std_dist = torch.norm(
                    (aligned - cull_mean) / cull_std, dim=-1
                )
                mask = std_dist < effective_cull_radius
                pts = pts[mask]

                if self.load_normals:
                    nrm = torch.from_numpy(npz[f"normals_{ci:03d}"])
                    nrm = nrm[mask]
                    normals_accum.append(nrm)
            else:
                if self.load_normals:
                    nrm = torch.from_numpy(npz[f"normals_{ci:03d}"])
                    normals_accum.append(nrm)

            points_accum.append(pts)

            if sum(p.shape[0] for p in points_accum) >= n_points:
                break

        surface_points = torch.cat(points_accum, dim=0)
        total = surface_points.shape[0]
        if total == 0:
            raise EmptyCullRegionError(
                f"No GT surface points inside cull radius "
                f"{effective_cull_radius} for scene {scene_info['scene_id']}."
            )
        if total > n_points:
            if np_rng is not None:
                sampled_idx = torch.from_numpy(np_rng.permutation(total)[:n_points])
            else:
                sampled_idx = torch.randperm(total)[:n_points]
        else:
            if np_rng is not None:
                sampled_idx = torch.from_numpy(np_rng.integers(0, total, size=(n_points,)))
            else:
                sampled_idx = torch.randint(0, total, (n_points,))
        surface_points = surface_points[sampled_idx]

        if self.load_normals:
            surface_normals = torch.cat(normals_accum, dim=0)[sampled_idx]
            return surface_points, surface_normals

        return surface_points

    # ------------------------------------------------------------------
    # Sample-file selection hook (overridable by subclasses)
    # ------------------------------------------------------------------

    def _select_sample_file(
        self,
        scene_info: dict,
        img_per_seq: Optional[int],
        item_rng: random.Random,
    ) -> str:
        """Return the path of the ``.pt`` cache file to load for this item.

        Default implementation: uniform random pick over
        ``scene_info["sample_files"]``. Subclasses can override to route to
        different caches depending on the requested view count
        (``img_per_seq``).
        """
        del img_per_seq  # unused in the default 16-view-only implementation
        return item_rng.choice(scene_info["sample_files"])

    # ------------------------------------------------------------------
    # Main data loading
    # ------------------------------------------------------------------

    def get_data(
        self,
        seq_index: int,
        img_per_seq: int,
        aspect_ratio: float = 1.0,
        n_points: int | None = None,
        sampler_index: int | None = None,
        **kwargs,
    ) -> dict:
        if sampler_index is None:
            sampler_index = seq_index
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        n_points = self.n_points if n_points is None else n_points
        scene_idx = seq_index % self.sequence_list_len
        scene_info = self.scenes[scene_idx]

        # Seed from (sampler_index, epoch).  ``sampler_index`` is the
        # original dataloader iteration index (unique per batch element),
        # so different iterations within the same epoch get different GT
        # point samples even when they happen to load the same scene.
        # When ``fixed_seed_surface_points`` is True the epoch term is
        # omitted so that the same scene always yields the same point
        # sample regardless of epoch (useful for validation consistency).
        _epoch = getattr(self, "epoch", 0)
        _seed = sampler_index * 100003 + (0 if self.fixed_seed_surface_points else _epoch * 12347)
        _item_rng = random.Random(_seed)
        _item_np_rng = np.random.default_rng(_seed)

        # ---- Load preprocessed VGGT cache ----
        # Subclasses can override ``_select_sample_file`` to dispatch on
        # ``img_per_seq`` (e.g. multiview variants that route to different
        # cache directories depending on the requested view count).
        sample_file = self._select_sample_file(
            scene_info, img_per_seq=img_per_seq, item_rng=_item_rng,
        )
        cached = torch.load(sample_file, map_location="cpu", weights_only=False)

        # ---- Load GT surface points (fresh random sample each call) ----
        scene_center = cached["scene_center"]
        scene_radius = cached["scene_radius"]
        alignment_L = cached["alignment_L"].squeeze(0)  # (3, 3)
        alignment_T = cached["alignment_T"].squeeze(0)  # (3,)

        # ---- Cast vggt_world_points to float32 to avoid precision issues and overflows ----
        if "vggt_world_points" in cached:
            cached["vggt_world_points"] = cached["vggt_world_points"].float()

        # ---- Sample per-scene cull_radius if configured as an interval ----
        # ``max_cull_radius`` is the ceiling used by the retry loop below:
        # if a sampled (or fixed) radius yields no GT points, we double it
        # up to this ceiling before giving up and letting BaseDataset
        # blacklist the scene.
        if self.cull_radius_range is not None:
            lo, hi = self.cull_radius_range
            if self._cull_curriculum_active:
                progress = min(
                    _epoch / self.cull_curriculum_warmup_epochs,
                    self.cull_curriculum_max_progress,
                )
                if self.cull_curriculum_law == "beta":
                    # Anneal alpha from alpha_init -> 1.0 over warmup_epochs,
                    # then derive beta so the mode stays fixed at
                    # ``cull_curriculum_mode``. At alpha=1.0 the formula
                    # yields beta=1.0, i.e. Beta(1, 1) = Uniform, so the
                    # curriculum smoothly relaxes to uniform sampling.
                    alpha_cur = self.cull_curriculum_alpha_init + (1.0 - self.cull_curriculum_alpha_init) * progress
                    mode_u = (self.cull_curriculum_mode - lo) / (hi - lo)
                    beta_cur = (alpha_cur * (1.0 - mode_u) + 2.0 * mode_u - 1.0) / mode_u
                    u = _item_rng.betavariate(alpha_cur, beta_cur)
                    effective_cull_radius = lo + u * (hi - lo)
                else:
                    # "power": power law peaked at ``cull_curriculum_mode``.
                    # k anneals k_init -> 1.0; at k=1 the distribution
                    # relaxes to Uniform(lo, hi).
                    #
                    # - Interior mode: two-sided power law. Each side is
                    #   a single-sided power law on its half-interval,
                    #   picked with probability proportional to the
                    #   half-interval's width, so the k=1 mixture is
                    #   exactly Uniform(lo, hi).
                    # - Mode at an endpoint: collapses to the classic
                    #   single-sided power law (``r = lo + v*(hi-lo)`` if
                    #   mode==lo, mirrored if mode==hi). Handled explicitly
                    #   to avoid an unused side-selection RNG draw.
                    k_cur = self.cull_curriculum_k_init + (1.0 - self.cull_curriculum_k_init) * progress
                    u = _item_rng.random()
                    v = u ** k_cur
                    left_width = self.cull_curriculum_mode - lo
                    right_width = hi - self.cull_curriculum_mode
                    total_width = hi - lo
                    if left_width <= 0.0:
                        effective_cull_radius = lo + v * total_width
                    elif right_width <= 0.0:
                        effective_cull_radius = hi - v * total_width
                    elif _item_rng.random() < left_width / total_width:
                        effective_cull_radius = self.cull_curriculum_mode - v * left_width
                    else:
                        effective_cull_radius = self.cull_curriculum_mode + v * right_width
            else:
                effective_cull_radius = _item_rng.uniform(lo, hi)
            max_cull_radius = hi
        elif self.cull_radius is not None:
            effective_cull_radius = self.cull_radius
            max_cull_radius = self.cull_radius
        else:
            effective_cull_radius = None
            max_cull_radius = None

        # ---- Compute per-scene culling stats if needed ----
        scene_mean, scene_std = None, None
        if self.per_scene_normalize and effective_cull_radius is not None:
            if "vggt_world_points" in cached:
                vggt_pts = cached["vggt_world_points"].squeeze(0)  # (N, H, W, 3)
                pts_flat = vggt_pts.reshape(-1, 3)
                if self.scene_normalize_mode == "median_dist_to_medianpoint":
                    scene_mean = pts_flat.median(dim=0).values  # (3,)
                else:
                    scene_mean = pts_flat.mean(dim=0)  # (3,)
                if self.scene_normalize_mode in ("median_dist_to_barycenter", "median_dist_to_medianpoint"):
                    dists = torch.norm(pts_flat - scene_mean, dim=-1)  # (N_flat,)
                    scene_std = dists.median().clamp(min=1e-6).expand(3)  # (3,) isotropic
                elif self.scene_normalize_mode == "std":
                    scene_std = pts_flat.std(dim=0).clamp(min=1e-6)  # (3,)
                else:
                    raise ValueError(f"Invalid scene_normalize_mode: {self.scene_normalize_mode}")

        # Retry loop: if the current cull radius yields no GT points, double
        # it (capped at ``max_cull_radius``) and try again. If even the max
        # radius yields nothing, propagate the exception so BaseDataset
        # blacklists this scene.
        while True:
            try:
                surface_data = self._load_surface_points(
                    scene_info, n_points,
                    alignment_L=alignment_L,
                    alignment_T=alignment_T,
                    scene_mean=scene_mean,
                    scene_std=scene_std,
                    np_rng=_item_np_rng,
                    effective_cull_radius=effective_cull_radius,
                )
                break
            except EmptyCullRegionError:
                if (
                    effective_cull_radius is None
                    or max_cull_radius is None
                    or effective_cull_radius >= max_cull_radius
                ):
                    raise
                new_radius = min(effective_cull_radius * 2.0, max_cull_radius)
                logging.warning(
                    f"[{scene_info['scene_id']}] Empty cull region at "
                    f"radius={effective_cull_radius:.3f}; retrying with "
                    f"radius={new_radius:.3f} (max={max_cull_radius:.3f})."
                )
                effective_cull_radius = new_radius
        if self.load_normals:
            surface_points, surface_normals = surface_data
        else:
            surface_points = surface_data

        # ---- Load denser GT points for chamfer evaluation (points only, no normals) ----
        chamfer_surface_points = None
        if self.chamfer_n_points is not None and self.chamfer_n_points > 0:
            # Use a separate RNG derived from the item RNG so the chamfer
            # point sample is independent from the training point sample
            # but still deterministic.
            _chamfer_np_rng = np.random.default_rng(_item_np_rng.integers(0, 2**31))
            chamfer_data = self._load_surface_points(
                scene_info, self.chamfer_n_points,
                alignment_L=alignment_L,
                alignment_T=alignment_T,
                scene_mean=scene_mean,
                scene_std=scene_std,
                np_rng=_chamfer_np_rng,
                effective_cull_radius=effective_cull_radius,
            )
            if self.load_normals:
                chamfer_surface_points, _ = chamfer_data
            else:
                chamfer_surface_points = chamfer_data

        # ---- Stack used token layers into a single tensor ----
        used_tokens = []
        for layer_idx in USED_TOKEN_LAYER_INDICES:
            token = cached["aggregated_tokens_list"][layer_idx]
            used_tokens.append(token.squeeze(0))  # (N, S, D)
        cached_aggregated_tokens = torch.stack(used_tokens, dim=0)  # (n_layers_used, N, S, D)

        # ---- Ensure consistent shapes ----
        if isinstance(scene_center, torch.Tensor) and scene_center.ndim == 1:
            scene_center = scene_center.unsqueeze(0)  # (1, 3)

        batch = {
            "seq_name": f"dl3dv_preprocessed_{scene_info['scene_id']}",
            "sample_file": sample_file,
            "ids": cached["ids"],
            "frame_num": len(cached["ids"]),
            "extrinsics": cached["colmap_extrinsics"],
            "intrinsics": cached["colmap_intrinsics"],
            "target_3d_points": surface_points,
            "scene_radius": scene_radius,
            "scene_center": scene_center,
            # Pre-computed VGGT outputs
            "cached_aggregated_tokens": cached_aggregated_tokens,
            "vggt_extrinsics": cached["vggt_extrinsics"].squeeze(0),
            "vggt_intrinsics": cached["vggt_intrinsics"].squeeze(0),
            "alignment_L": cached["alignment_L"].squeeze(0),
            "alignment_T": cached["alignment_T"].squeeze(0),
        }

        if chamfer_surface_points is not None:
            batch["chamfer_target_3d_points"] = chamfer_surface_points

        if self.load_normals:
            batch["target_normals"] = surface_normals

        if effective_cull_radius is not None:
            batch["cull_radius"] = torch.tensor(effective_cull_radius, dtype=torch.float32)

        if "vggt_world_points" in cached:
            batch["vggt_world_points"] = cached["vggt_world_points"].squeeze(0)

        if "rgb_images" in cached:
            batch["rgb_images"] = cached["rgb_images"].squeeze(0)  # (N, H, W, 3)

        return batch
