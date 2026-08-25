"""Distributed rank helpers."""
import os


def get_machine_local_and_dist_rank():
    """Return ``(local_rank, distributed_rank)`` from the environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", None))
    distributed_rank = int(os.environ.get("RANK", None))
    assert (
        local_rank is not None and distributed_rank is not None
    ), "Please set the RANK and LOCAL_RANK environment variables."
    return local_rank, distributed_rank
