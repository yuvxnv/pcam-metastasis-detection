"""PCam data loading. Works on a Mac (MPS/CPU) and on Colab (CUDA).

PCam images live in big .h5 files. We read them lazily (one image at a time)
instead of loading everything into RAM, because the Mac has only 8 GB.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# torchvision's PCAM names its splits train/val/test; the files say "valid".
_FILE_SPLIT = {"train": "train", "val": "valid", "test": "test"}


def h5_paths(root, split):
    base = f"{root}/pcam/camelyonpatch_level_2_split_{_FILE_SPLIT[split]}"
    return base + "_x.h5", base + "_y.h5"


def load_labels(root, split):
    """Labels are tiny (1 byte each), so loading all of them is fine."""
    _, ypath = h5_paths(root, split)
    with h5py.File(ypath, "r") as f:
        return f["y"][:].reshape(-1).astype(np.int64)


class PCamH5(Dataset):
    """Reads patches straight from the .h5 file. `indices` lets us use a subsample."""

    def __init__(self, root, split, indices=None, transform=None):
        self.xpath, ypath = h5_paths(root, split)
        self.labels = load_labels(root, split)
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)
        self.transform = transform
        self._file = None  # opened lazily so each DataLoader worker gets its own handle

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._file is None:
            self._file = h5py.File(self.xpath, "r")
        j = int(self.indices[i])
        img = self._file["x"][j]  # (96, 96, 3) uint8
        label = int(self.labels[j])
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> (3, 96, 96) uint8
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def stratified_subsample(labels, n, seed):
    """Pick n indices keeping the class ratio. Same seed -> same subsample."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(labels):
        c_idx = np.flatnonzero(labels == c)
        k = round(n * len(c_idx) / len(labels))
        idx.append(rng.choice(c_idx, size=k, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return np.sort(idx)  # sorted = faster sequential h5 reads
