"""Console + file logging setup (file handler on rank 0 only)."""
import atexit
import copy
import functools
import logging
import sys

from iopath.common.file_io import g_pathmgr

from .general import safe_makedirs


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # Cache the opened file object so repeated calls with the same filename
    # write to the same handle.
    log_buffer_kb = 1 * 1024  # 1KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
    all_ranks: bool = False,
):
    """Set up stdout + optional file logging. File logging is rank-0 only
    (unless ``all_ranks`` is set)."""
    global LOGGING_STATE
    LOGGING_STATE = copy.deepcopy(locals())

    log_filename = None
    if output_dir:
        safe_makedirs(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"
        elif all_ranks:
            log_filename = f"{output_dir}/log_{rank}.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []
    logging.root.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)
    logger.addHandler(console_handler)

    if log_filename is not None:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger
