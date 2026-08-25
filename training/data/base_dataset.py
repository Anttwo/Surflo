"""Minimal base dataset for the preprocessed (cached) training path.

The original ``BaseDataset`` also handled raw-image loading / resizing /
augmentation (``process_one_image``, ``get_nearby_ids`` and the ``dataset_util``
image ops). None of that runs on the preprocessed path (batches carry cached
VGGT tokens, never raw ``images``), so only the retry-on-failure ``__getitem__``
loop is kept here.
"""
import logging
import random

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Abstract dataset with a robust retry loop around :meth:`get_data`.

    Concrete datasets implement :meth:`get_data` and set ``len_train``,
    ``sequence_list`` and ``sequence_list_len``.
    """

    def __init__(self, common_conf):
        super().__init__()
        # ``common_conf`` fields are consumed by the concrete dataset; nothing
        # is needed at the base level for the preprocessed path.
        del common_conf

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        """Fetch one item, retrying on a different random scene upon failure.

        Args:
            idx_N: ``(seq_index, img_per_seq, aspect_ratio)`` or
                ``(seq_index, img_per_seq, aspect_ratio, sampler_index)``. The
                optional 4th element is the dataloader sampler index, forwarded
                to :meth:`get_data` for RNG seeding.
        """
        if len(idx_N) >= 4:
            seq_index, img_per_seq, aspect_ratio, sampler_index = idx_N[:4]
        else:
            seq_index, img_per_seq, aspect_ratio = idx_N
            sampler_index = seq_index

        max_retries = getattr(self, "max_retries_per_getitem", 10)
        failed_scenes = getattr(self, "failed_scenes", set())

        for retry in range(max_retries):
            try:
                return self.get_data(
                    seq_index=seq_index, img_per_seq=img_per_seq,
                    aspect_ratio=aspect_ratio, sampler_index=sampler_index,
                )
            except Exception as e:
                logging.warning(
                    f"Error loading item at seq_index={seq_index}, img_per_seq={img_per_seq}, "
                    f"aspect_ratio={aspect_ratio} (retry {retry + 1}/{max_retries}): "
                    f"{type(e).__name__}: {str(e)}"
                )

                if hasattr(self, "sequence_list") and seq_index < len(self.sequence_list):
                    failed_scenes.add(self.sequence_list[seq_index])

                if retry >= max_retries - 1:
                    logging.error(f"Failed to load data after {max_retries} retries. Giving up.")
                    raise

                # Try a different, not-yet-failed random scene.
                if hasattr(self, "sequence_list_len"):
                    for _ in range(100):
                        new_seq_index = random.randint(0, self.sequence_list_len - 1)
                        if hasattr(self, "sequence_list"):
                            if self.sequence_list[new_seq_index] not in failed_scenes:
                                seq_index = new_seq_index
                                break
                    else:
                        seq_index = random.randint(0, self.sequence_list_len - 1)
                else:
                    seq_index = (seq_index + 1) % len(self)

        raise RuntimeError(f"Failed to load any data after {max_retries} retries")

    def get_data(self, seq_index, img_per_seq, seq_name=None, ids=None,
                 aspect_ratio=1.0, sampler_index=None, **kwargs):
        raise NotImplementedError(
            "get_data must be implemented by the concrete dataset subclass."
        )
