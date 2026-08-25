"""Dynamic dataloader: per-iteration view-count / aspect-ratio sampling.

Each iteration draws a single view count ``N`` and aspect ratio, shared by
every scene in the batch, and sizes the batch as ``floor(max_img_per_gpu / N)``.
All DDP ranks draw the same ``N`` at the same step (identical seeding), so
global batches are homogeneous in ``N``.
"""
import random
from abc import ABC
from typing import Callable, Optional

import numpy as np
from hydra.utils import instantiate
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .worker_fn import get_worker_init_fn


def _debug_collate(batch):
    """``default_collate`` wrapper that reports which key breaks collation."""
    import torch
    from torch.utils.data._utils.collate import default_collate

    if not isinstance(batch, list) or not batch or not isinstance(batch[0], dict):
        return default_collate(batch)

    keys = list(batch[0].keys())
    result = {}
    for key in keys:
        vals = [d[key] for d in batch]
        try:
            result[key] = default_collate(vals)
        except Exception as e:
            types_and_shapes = []
            for i, v in enumerate(vals):
                if isinstance(v, torch.Tensor):
                    types_and_shapes.append(
                        f"  [{i}] Tensor dtype={v.dtype} shape={v.shape} "
                        f"is_contiguous={v.is_contiguous()}"
                    )
                else:
                    types_and_shapes.append(f"  [{i}] {type(v).__name__} = {repr(v)[:200]}")
            detail = "\n".join(types_and_shapes)
            scenes = [d.get("seq_name", "?") for d in batch]
            raise RuntimeError(
                f"Collation failed on key='{key}' with {len(vals)} items, scenes={scenes}:\n{detail}"
            ) from e
    return result


class DynamicTorchDataset(ABC):
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

        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

        self.aspect_ratio_range = common_config.augs.aspects  # e.g. [0.5, 1.0]
        self.image_num_range = common_config.img_nums          # e.g. [2, 24]

        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        self.sampler = DynamicDistributedSampler(self.dataset, seed=seed, shuffle=shuffle)
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.image_num_range,
            seed=seed,
            max_img_per_gpu=max_img_per_gpu,
        )

    def get_loader(self, epoch, num_workers=None):
        print("Building dynamic dataloader with epoch:", epoch)

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


class DynamicBatchSampler(Sampler):
    """Draws a shared (view count, aspect ratio) per iteration and yields a
    batch of ``floor(max_img_per_gpu / N)`` items."""

    def __init__(self, sampler, aspect_ratio_range, image_num_range,
                 epoch=0, seed=42, max_img_per_gpu=48):
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.seed = int(seed)
        self.rng = random.Random()
        self.np_rng = np.random.default_rng(self.seed)

        self.image_num_weights = {n: 1.0 for n in range(image_num_range[0], image_num_range[1] + 1)}
        self.possible_nums = np.array(
            [n for n in self.image_num_weights.keys() if self.image_num_range[0] <= n <= self.image_num_range[1]]
        )
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        self.max_img_per_gpu = max_img_per_gpu
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
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

                batch_size = np.floor(self.max_img_per_gpu / random_image_num).astype(int)
                batch_size = max(1, batch_size)

                current_batch = []
                for _ in range(batch_size):
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


class DynamicDistributedSampler(DistributedSampler):
    """``DistributedSampler`` that attaches the current (image_num, aspect_ratio)
    to each yielded index."""

    def __init__(self, dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = False, seed: int = 0, drop_last: bool = False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.aspect_ratio = None
        self.image_num = None

    def __iter__(self):
        for idx in super().__iter__():
            yield (idx, self.image_num, self.aspect_ratio)

    def update_parameters(self, aspect_ratio, image_num):
        self.aspect_ratio = aspect_ratio
        self.image_num = image_num
