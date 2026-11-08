"""Download PCam with torchvision, one split at a time (valid is small, so it
tests the Google Drive connection before we commit to the 6 GB train file).
Usage: python scripts/download_data.py [--root data]

Why the patch below: torchvision unpacks .gz files by reading the WHOLE file into
RAM (the unpacked train file is 7.2 GB). On a free Colab machine that gets the
process killed silently. We swap in a version that streams in 16 MB chunks.
"""
import argparse
import gzip
import os
import shutil

import torchvision.datasets.pcam as _pcam
from torchvision.datasets import PCAM


def _stream_decompress(from_path, to_path=None, remove_finished=False):
    to_path = to_path or from_path[: -len(".gz")]
    with gzip.open(from_path, "rb") as src, open(to_path, "wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    if remove_finished:
        os.remove(from_path)
    return to_path


_pcam._decompress = _stream_decompress  # pcam.py looks this name up when unpacking

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    args = p.parse_args()

    for split in ["val", "test", "train"]:
        print(f"=== downloading {split} === (unpacking after the download is silent and can take several minutes; do not interrupt)", flush=True)
        ds = PCAM(root=args.root, split=split, download=True)
        print(f"{split}: {len(ds)} images OK", flush=True)
