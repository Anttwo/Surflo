"""Compose one or more leaf datasets behind a tuple-indexable interface.

Trimmed to the preprocessed path: the original also applied color
augmentation and point-track generation, but preprocessed batches never carry
raw ``images`` (only cached VGGT tokens), so that machinery (and its
``augmentation`` / ``track_util`` deps) is dropped here.
"""
import bisect
import random
from abc import ABC

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset


class ComposedDataset(Dataset, ABC):
    """Instantiate leaf datasets with a shared ``common_config`` and expose
    them through a single tuple-indexable dataset."""

    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        base_dataset_list = []
        for baseset_dict in dataset_configs:
            base_dataset_list.append(instantiate(baseset_dict, common_conf=common_config))

        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        if any(getattr(ds, "inside_random", True) is False for ds in base_dataset_list):
            self.base_dataset.inside_random = False

        # Optional fixed overrides (debugging): force a view count / aspect ratio.
        self.fixed_num_images = common_config.fix_img_num
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        self.training = common_config.training
        self.common_config = common_config

        self.total_samples = len(self.base_dataset)

    @property
    def epoch(self):
        return getattr(self, "_epoch", 0)

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        self.base_dataset.epoch = value

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx_tuple):
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio) + idx_tuple[3:]

        batch = self.base_dataset[idx_tuple]
        seq_name = batch["seq_name"]

        # Convert numpy arrays to tensors (skip values that are already tensors).
        for key in ["depths", "extrinsics", "intrinsics", "cam_points", "world_points"]:
            if key in batch and not isinstance(batch[key], torch.Tensor):
                batch[key] = torch.from_numpy(np.stack(batch[key]).astype(np.float32))

        if not isinstance(batch["ids"], torch.Tensor):
            batch["ids"] = torch.from_numpy(batch["ids"])

        batch["seq_name"] = seq_name
        return batch


class TupleConcatDataset(ConcatDataset):
    """``ConcatDataset`` that accepts a tuple index ``(sample_idx, N, aspect)``.

    The first element selects the sample across the concatenated datasets; the
    full tuple (plus the original sampler index appended as a 4th element for
    RNG seeding) is forwarded to the selected dataset.
    """

    def __init__(self, datasets, common_config):
        super().__init__(datasets)
        self.inside_random = common_config.inside_random

    @property
    def epoch(self):
        return getattr(self, "_epoch", 0)

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        for ds in self.datasets:
            ds.epoch = value

    def __getitem__(self, idx):
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]

        original_sampler_idx = idx

        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:] + (original_sampler_idx,)
        else:
            raise ValueError("Tuple index must have exactly three elements")

        return self.datasets[dataset_idx][idx_tuple]
