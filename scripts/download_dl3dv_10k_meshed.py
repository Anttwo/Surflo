#!/usr/bin/env python3
"""Download DL3DV-10K-Meshed, one subset at a time, with retries.

Requires access to the gated repo. Authenticate first with `hf auth login`.

Examples
--------
    python download_dl3dv_10k_meshed.py --odir dl3dv-10k-meshed                 # everything
    python download_dl3dv_10k_meshed.py --odir dl3dv-10k-meshed --subset 1K 2K  # selected subsets
    python download_dl3dv_10k_meshed.py --odir dl3dv-10k-meshed --workers 4     # throttle if rate-limited
"""

import argparse
import sys
import time

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "AntoineGuedon/DL3DV-10K-Meshed"
SUBSETS = ["1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K", "11K"]


def download_subset(subset, odir, workers, max_retries, backoff):
    """Fetch one subset, retrying with exponential backoff on transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=f"{subset}/*.tar",
                local_dir=odir,
                max_workers=workers,
            )
            return True
        except (HfHubHTTPError, OSError, ConnectionError) as err:
            if attempt == max_retries:
                print(f"[{subset}] giving up after {max_retries} attempts: {err}")
                return False
            wait = backoff * 2 ** (attempt - 1)
            print(f"[{subset}] attempt {attempt} failed ({err}); retrying in {wait}s")
            time.sleep(wait)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odir", required=True, help="output directory")
    parser.add_argument("--subset", nargs="+", choices=SUBSETS, default=SUBSETS,
                        help="subsets to download (default: all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel downloads; lower this if you hit HTTP 429")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--backoff", type=int, default=30,
                        help="initial retry delay in seconds")
    args = parser.parse_args()

    failed = []
    for subset in args.subset:
        print(f"=== {subset} ===")
        if not download_subset(subset, args.odir, args.workers,
                               args.max_retries, args.backoff):
            failed.append(subset)

    if failed:
        print(f"\nIncomplete: {', '.join(failed)}. Re-run to resume.")
        sys.exit(1)
    print("\nAll requested subsets downloaded.")


if __name__ == "__main__":
    main()