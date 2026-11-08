"""Train a model on PCam. Uses TRAIN for learning and VALIDATION for picking the
best epoch. The test set is never loaded here (see src/evaluate.py).

Examples
  Mac, quick dev run on a subsample:
    python -m src.train --model cnn --n-train 20000 --n-val 4000 --epochs 5
  Colab, full run:
    python -m src.train --model resnet18 --epochs 10
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.data import PCamH5, load_labels, stratified_subsample
from src.models import build_model

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats (used for both models)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_transform():
    """Augmentation: TRAINING ONLY. Tissue has no 'up', so flips and 90-degree
    rotations are label-safe. Color jitter is kept mild (stain colour varies between slides)."""
    return v2.Compose([
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
        v2.RandomChoice([v2.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD),
    ])


def eval_transform():
    """No augmentation for validation/test - only the same normalisation."""
    return v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu())
    return torch.cat(probs).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "resnet18"], required=True)
    p.add_argument("--root", default="data")
    p.add_argument("--n-train", type=int, default=None, help="subsample size (default: all)")
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="", help="suffix for output names, e.g. 'dev'")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    name = args.model + (f"_{args.tag}" if args.tag else "")
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    print(f"device={device}  model={args.model}  seed={args.seed}", flush=True)

    ytr, yva = load_labels(args.root, "train"), load_labels(args.root, "val")
    tr_idx = stratified_subsample(ytr, args.n_train, args.seed) if args.n_train else None
    va_idx = stratified_subsample(yva, args.n_val, args.seed) if args.n_val else None
    train_ds = PCamH5(args.root, "train", tr_idx, train_transform())
    val_ds = PCamH5(args.root, "val", va_idx, eval_transform())
    print(f"train={len(train_ds)}  val={len(val_ds)}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, generator=g, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, 256, shuffle=False, **kw)

    model = build_model(args.model).to(device)
    # ResNet gets a gentler learning rate: its pretrained weights shouldn't be wrecked.
    lr = args.lr if args.model == "cnn" else min(args.lr, 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        sched.step()

        vp = predict(model, val_loader, device)
        vy = val_ds.labels[val_ds.indices]
        val_auc = roc_auc_score(vy, vp)
        val_acc = float(((vp > 0.5) == vy).mean())
        history.append({"epoch": epoch, "train_loss": run_loss / n, "val_auc": val_auc,
                        "val_acc": val_acc, "seconds": time.time() - t0})
        print(history[-1], flush=True)

        if val_auc > best_auc:  # best epoch chosen on VALIDATION AUC only
            best_auc = val_auc
            torch.save(model.state_dict(), f"checkpoints/{name}.pt")
            best_epoch = epoch

    pd.DataFrame(history).to_csv(f"results/history_{name}.csv", index=False)
    json.dump({"best_epoch": best_epoch, "best_val_auc": best_auc, "args": vars(args)},
              open(f"results/best_{name}.json", "w"), indent=2)
    print(f"done. best epoch={best_epoch} val_auc={best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
