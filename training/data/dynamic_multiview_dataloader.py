"""Multi-view variant of :class:`DynamicTorchDataset`.

Two differences vs the standard dynamic dataloader:

* The per-iter batch size ``B`` is **fixed** to ``max_img_per_gpu //
  n_max_views`` regardless of the sampled view count ``N`` (downstream buffers
  scale with the number of scenes ``B``, not images).
* ``image_num_weights`` gives each view count ``N`` an **unnormalized
  probability**. It defaults to ``{}`` = uniform over ``img_nums``; any ``N``
  the mapping omits gets ``1.0``, and ``0.0`` removes that ``N`` from the draw
  entirely. ``{16: 2}`` makes ``N=16`` twice as likely as every other count.
  The weight is per ``N``, not per cache file: ``N`` is drawn first, then a
  file is picked uniformly among that ``N``'s caches.

Reuses :class:`DynamicDistributedSampler` so the ``(idx, image_num,
aspect_ratio)`` plumbing is unchanged, and seeds every draw with ``seed +
epoch * 100`` so all DDP ranks agree on ``N`` at each step.
"""
import random
from typing import Callable, Optional

import numpy as np
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Sampler

from .dynamic_dataloader import DynamicDistributedSampler, _debug_collate
from .worker_fn import get_worker_init_fn


class MultiviewDynamicTorchDataset:
    """Dynamic dataloader wrapper for the multi-view dataset."""

    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
        max_img_per_gpu: int = 48,
        n_max_views: int = 16,
        image_num_weights: Optional[dict] = None,
    ) -> None:
        self.dataset_config = dataset
        self.common_config = common_config
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.max_img_per_gpu = max_img_per_gpu
        self.n_max_views = int(n_max_views)

        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

        self.aspect_ratio_range = common_config.augs.aspects
        self.image_num_range = common_config.img_nums

        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")
        if (
            len(self.image_num_range) != 2
            or self.image_num_range[0] < 1
            or self.image_num_range[0] > self.image_num_range[1]
            or self.image_num_range[1] > self.n_max_views
        ):
            raise ValueError(
                f"image_num_range must be [min, max] with "
                f"1 <= min <= max <= n_max_views ({self.n_max_views}), got {self.image_num_range}"
            )

        self._check_view_count_coverage(image_num_weights)

        self.sampler = DynamicDistributedSampler(self.dataset, seed=seed, shuffle=shuffle)
        self.batch_sampler = MultiviewDynamicBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.image_num_range,
            seed=seed,
            max_img_per_gpu=max_img_per_gpu,
            n_max_views=self.n_max_views,
            image_num_weights=image_num_weights,
        )

    def _check_view_count_coverage(self, image_num_weights: Optional[dict]) -> None:
        """Fail at construction if a *selectable* view count has no cache anywhere.

        Without this the gap only surfaces mid-epoch, the first time the sampler
        happens to draw that ``N``. View counts whose weight is ``0`` are skipped:
        they are never drawn, so they need not be preprocessed. Leaf datasets that
        do not expose ``view_count_coverage`` (anything other than the preprocessed
        DL3DV datasets) are skipped.

        Every leaf must cover every selectable ``N``: the sampler draws one ``N``
        for the whole batch, but individual items may come from any leaf.

        Two failure modes, handled differently:

        * **No scene at all** serves some selectable ``N`` -> raise, since no
          amount of pruning would make that ``N`` drawable.
        * **Some** scenes serve it and others do not -> drop the ones that do
          not (with a warning), so what remains is usable for every ``N`` the
          sampler can draw. Without this the gap only surfaces mid-epoch, the
          first time that ``N`` is drawn for a scene lacking it.
        """
        base = getattr(self.dataset, "base_dataset", None)
        leaves = getattr(base, "datasets", None) or []
        lo, hi = int(self.image_num_range[0]), int(self.image_num_range[1])
        weights = {int(k): float(v) for k, v in dict(image_num_weights or {}).items()}
        requested = {n for n in range(lo, hi + 1) if weights.get(n, 1.0) > 0.0}

        pruned_any = False
        for leaf in leaves:
            coverage = getattr(leaf, "view_count_coverage", None)
            if coverage is None:
                continue
            missing = sorted(n for n in requested if not coverage.get(n))
            if missing:
                raise RuntimeError(
                    f"{type(leaf).__name__} has no preprocessed cache for view "
                    f"count(s) {missing}, but img_nums={list(self.image_num_range)} "
                    f"requests them. Scenes per view count: "
                    f"{ {n: c for n, c in coverage.items() if c} }. "
                    f"Preprocess the missing view counts, narrow img_nums, or give "
                    f"them weight 0 in image_num_weights."
                )
            restrict = getattr(leaf, "restrict_to_view_counts", None)
            if restrict is not None:
                _, n_dropped = restrict(requested)
                pruned_any = pruned_any or n_dropped > 0

        if pruned_any:
            # ConcatDataset caches cumulative leaf lengths at construction, and
            # ComposedDataset caches the total. Both are stale once a leaf whose
            # length tracks its scene count (the val split) has been pruned.
            if hasattr(base, "cumulative_sizes"):
                base.cumulative_sizes = base.cumsum(base.datasets)
            if hasattr(self.dataset, "total_samples"):
                self.dataset.total_samples = len(base)

    def get_loader(self, epoch, num_workers=None):
        print(
            f"Building multiview dynamic dataloader with epoch={epoch}, "
            f"fixed batch_size={self.batch_sampler.batch_size}, "
            f"image_num_range={list(self.image_num_range)}, "
            f"image_num_weights={self.batch_sampler.image_num_weights}"
        )

        self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        effective_num_workers = self.num_workers if num_workers is None else num_workers
        effective_persistent = self.persistent_workers and effective_num_workers > 0

        collate = self.collate_fn if self.collate_fn is not None else _debug_collate
        return DataLoader(
            self.dataset,
            num_workers=effective_num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=collate,
            persistent_workers=effective_persistent,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=effective_num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )


class MultiviewDynamicBatchSampler(Sampler):
    """Like :class:`DynamicBatchSampler` but with a FIXED per-iter batch size of
    ``floor(max_img_per_gpu / n_max_views)`` and view-count weights."""

    def __init__(self, sampler, aspect_ratio_range, image_num_range,
                 epoch: int = 0, seed: int = 42, max_img_per_gpu: int = 48,
                 n_max_views: int = 16, image_num_weights: Optional[dict] = None):
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.seed = int(seed)
        self.rng = random.Random()
        self.np_rng = np.random.default_rng(self.seed)

        self.n_max_views = int(n_max_views)
        self.max_img_per_gpu = int(max_img_per_gpu)
        self.batch_size = max(1, int(np.floor(self.max_img_per_gpu / self.n_max_views)))

        # Per-N unnormalized probabilities. The mapping may be partial: any N
        # in range that it omits gets 1.0, so ``{}`` (the default) is uniform
        # and ``{16: 2}`` doubles N=16 while leaving the rest alone. A weight
        # of 0.0 drops that N from the draw.
        provided = {int(k): float(v) for k, v in dict(image_num_weights or {}).items()}
        in_range = set(range(image_num_range[0], image_num_range[1] + 1))
        # Out-of-range keys are almost always a typo (``{6: 2}`` for
        # ``{16: 2}``); silently ignoring them would apply the wrong schedule.
        stray = sorted(k for k in provided if k not in in_range)
        if stray:
            raise ValueError(
                f"image_num_weights has key(s) {stray} outside "
                f"img_nums={list(image_num_range)}. Remove them or widen img_nums."
            )
        if any(v < 0.0 for v in provided.values()):
            raise ValueError(f"image_num_weights must be >= 0, got {provided}")
        self.image_num_weights = {n: provided.get(n, 1.0) for n in sorted(in_range)}

        self.possible_nums = np.array(sorted(self.image_num_weights.keys()))
        weights = np.array([self.image_num_weights[int(n)] for n in self.possible_nums], dtype=np.float64)
        if weights.sum() <= 0.0:
            raise ValueError(
                f"image_num_weights must sum to > 0 (at least one view count has to "
                f"be selectable), got {self.image_num_weights}"
            )
        self.normalized_weights = weights / weights.sum()

        self.set_epoch(epoch)

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        epoch_seed = self.seed + int(epoch) * 100
        self.rng.seed(epoch_seed)
        self.np_rng = np.random.default_rng(epoch_seed)

    def __iter__(self):
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                random_image_num = int(self.np_rng.choice(self.possible_nums, p=self.normalized_weights))
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)
                self.sampler.update_parameters(aspect_ratio=random_aspect_ratio, image_num=random_image_num)

                current_batch = []
                for _ in range(self.batch_size):
                    try:
                        current_batch.append(next(sampler_iterator))
                    except StopIteration:
                        break

                if not current_batch:
                    break

                yield current_batch
            except StopIteration:
                break

    def __len__(self):
        return 1000000
