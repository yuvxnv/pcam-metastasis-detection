"""Download PCam with torchvision, one split at a time (valid is small, so it
tests the Google Drive connection before we commit to the 6 GB train file).
Usage: python scripts/download_data.py [--root data]
"""
import argparse
from torchvision.datasets import PCAM

p = argparse.ArgumentParser()
p.add_argument("--root", default="data")
args = p.parse_args()

for split in ["val", "test", "train"]:
    print(f"=== downloading {split} ===", flush=True)
    ds = PCAM(root=args.root, split=split, download=True)
    print(f"{split}: {len(ds)} images OK", flush=True)
