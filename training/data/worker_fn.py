"""Distributed helpers and deterministic per-worker seeding for dataloaders."""
import random
from functools import partial

import numpy as np
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def default_worker_init_fn(worker_id, num_workers, epoch, seed=0):
    """Seed each dataloader worker uniquely across rank / worker / epoch."""
    rank = get_rank()
    world_size = get_world_size()

    RANK_MULTIPLIER = 1
    WORKER_MULTIPLIER = 1
    WORLD_MULTIPLIER = 1
    EPOCH_MULTIPLIER = 12345

    worker_seed = (
        rank * num_workers * RANK_MULTIPLIER
        + worker_id * WORKER_MULTIPLIER
        + seed
        + world_size * WORLD_MULTIPLIER
        + epoch * EPOCH_MULTIPLIER
    )

    torch.random.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_worker_init_fn(seed, num_workers, epoch, worker_init_fn=None):
    """Return a ``worker_init_fn`` for :class:`torch.utils.data.DataLoader`."""
    if worker_init_fn is not None:
        return worker_init_fn

    return partial(
        default_worker_init_fn,
        num_workers=num_workers,
        epoch=epoch,
        seed=seed,
    )
