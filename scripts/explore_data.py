"""Step 1: class balance per split + grid of sample patches per class.
Usage: python scripts/explore_data.py [--root data]
"""
import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import h5_paths, load_labels

p = argparse.ArgumentParser()
p.add_argument("--root", default="data")
args = p.parse_args()
Path("results").mkdir(exist_ok=True)

rows = []
for split in ["train", "val", "test"]:
    y = load_labels(args.root, split)
    rows.append({"split": split, "n": len(y), "negative(0)": int((y == 0).sum()),
                 "positive(1)": int((y == 1).sum()), "pct_positive": round(100 * y.mean(), 2)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("results/class_balance.csv", index=False)

# Sample grid: 2 rows (class 0 / class 1) x 10 random patches, from the training split
rng = np.random.default_rng(0)
y = load_labels(args.root, "train")
xpath, _ = h5_paths(args.root, "train")
fig, axes = plt.subplots(2, 10, figsize=(20, 4.6))
with h5py.File(xpath, "r") as f:
    for r, c in enumerate([0, 1]):
        idx = np.sort(rng.choice(np.flatnonzero(y == c), 10, replace=False))
        for k, j in enumerate(idx):
            axes[r, k].imshow(f["x"][int(j)])
            axes[r, k].axis("off")
        axes[r, 0].set_title("NO tumor in center" if c == 0 else "TUMOR in center",
                             loc="left", fontsize=12)
plt.tight_layout()
plt.savefig("results/sample_patches.png", dpi=120)
print("saved results/class_balance.csv and results/sample_patches.png")
