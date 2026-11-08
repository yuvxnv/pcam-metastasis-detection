"""Step 3: evaluate a trained model; save metrics CSV + plots to results/.

Dev (validation set, as often as you like):
  python -m src.evaluate --model cnn --tag dev --split val --n 4000
Final (test set, ONCE per model - a lock file stops accidental re-runs):
  python -m src.evaluate --model resnet18 --split test --final-test

The decision threshold is fixed at 0.5 in advance. It is NOT tuned on the test set.
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.data import PCamH5, h5_paths, stratified_subsample, load_labels
from src.models import build_model
from src.train import MEAN, STD, eval_transform, get_device, predict, set_seed

THRESHOLD = 0.5


def bootstrap_auc_ci(y, p, n_boot=2000, seed=0):
    """Resample the test patches with replacement many times, recompute AUC each time;
    the 2.5th and 97.5th percentiles give a 95% confidence interval."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return ece


def read_patches(root, split, idx):
    xpath, _ = h5_paths(root, split)
    with h5py.File(xpath, "r") as f:
        return [f["x"][int(j)] for j in idx]


def grad_cam(model, target_layer, img_u8, device):
    """Grad-CAM: which spatial regions pushed the 'tumor' score up?
    Take the last conv feature maps, weight each map by how much the score's gradient
    flows through it, sum, keep positives, upsample to 96x96."""
    tf = eval_transform()
    x = tf(torch.from_numpy(img_u8).permute(2, 0, 1)).unsqueeze(0).to(device)
    acts = {}
    h = target_layer.register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    model.eval()
    logit = model(x)
    h.remove()
    a = acts["a"]
    (g,) = torch.autograd.grad(logit.sum(), a)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(96, 96), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    return cam / (cam.max() + 1e-8), torch.sigmoid(logit).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "resnet18"], required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--root", default="data")
    p.add_argument("--tag", default="")
    p.add_argument("--n", type=int, default=None, help="subsample size (dev only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--final-test", action="store_true", help="required to score the test set")
    p.add_argument("--no-gradcam", action="store_true")
    args = p.parse_args()

    name = args.model + (f"_{args.tag}" if args.tag else "")
    out = f"{name}_{args.split}"
    Path("results").mkdir(exist_ok=True)

    if args.split == "test":
        lock = Path(f"results/.test_used_{name}")
        if not args.final_test:
            raise SystemExit("Refusing: scoring the test set needs --final-test.")
        if args.n:
            raise SystemExit("Refusing: the test set must be scored in full (no --n).")
        if lock.exists():
            raise SystemExit(f"Refusing: test set already scored for '{name}' ({lock}). "
                             "Scoring it repeatedly and picking favourites defeats its purpose.")
        lock.write_text("test set scored once for this model\n")

    set_seed(args.seed)
    device = get_device()
    labels_all = load_labels(args.root, args.split)
    idx = stratified_subsample(labels_all, args.n, args.seed) if args.n else None
    ds = PCamH5(args.root, args.split, idx, eval_transform())
    y = ds.labels[ds.indices]
    loader = DataLoader(ds, 256, shuffle=False, num_workers=2)

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{name}.pt", map_location=device))
    prob = predict(model, loader, device)
    print(f"scored {len(y)} {args.split} patches with {name}")

    # ---- metrics ----
    pred = (prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    auc = roc_auc_score(y, prob)
    lo, hi = bootstrap_auc_ci(y, prob, seed=args.seed)
    row = {"model": name, "split": args.split, "n": len(y), "auc": auc,
           "auc_ci95_low": lo, "auc_ci95_high": hi,
           "accuracy": (tp + tn) / len(y), "sensitivity": tp / (tp + fn),
           "specificity": tn / (tn + fp), "ece": expected_calibration_error(y, prob),
           "tn": tn, "fp": fp, "fn": fn, "tp": tp, "threshold": THRESHOLD}
    print(pd.Series(row).to_string())

    mpath = Path("results/metrics.csv")
    new = pd.DataFrame([row])
    if mpath.exists():
        old = pd.read_csv(mpath)
        old = old[~((old.model == name) & (old.split == args.split))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(mpath, index=False)
    pd.DataFrame({"index": ds.indices, "label": y, "prob": prob}).to_csv(
        f"results/preds_{out}.csv", index=False)

    # ---- ROC ----
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    ax.scatter([1 - row["specificity"]], [row["sensitivity"]], color="red", zorder=3,
               label=f"threshold {THRESHOLD}")
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
           title=f"ROC - {name} ({args.split})")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"results/roc_{out}.png", dpi=130); plt.close(fig)

    # ---- confusion matrix ----
    fig, ax = plt.subplots(figsize=(4.4, 4))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred normal", "pred tumor"],
           yticklabels=["true normal", "true tumor"], title=f"Confusion matrix - {name} ({args.split})")
    fig.tight_layout(); fig.savefig(f"results/confusion_{out}.png", dpi=130); plt.close(fig)

    # ---- calibration ----
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", lw=2, label=f"{name} (ECE {row['ece']:.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction of tumor patches",
           title=f"Calibration - {name} ({args.split})")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"results/calibration_{out}.png", dpi=130); plt.close(fig)

    # ---- 12 misclassified patches: 6 false negatives + 6 false positives, random (seeded) ----
    rng = np.random.default_rng(args.seed)
    fn_pos = np.flatnonzero((y == 1) & (pred == 0))
    fp_pos = np.flatnonzero((y == 0) & (pred == 1))
    pick_fn = rng.choice(fn_pos, min(6, len(fn_pos)), replace=False)
    pick_fp = rng.choice(fp_pos, min(6, len(fp_pos)), replace=False)
    picks = [(k, "MISSED tumor (FN)") for k in pick_fn] + [(k, "FALSE ALARM (FP)") for k in pick_fp]
    imgs = read_patches(args.root, args.split, [ds.indices[k] for k, _ in picks])
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
    for ax, (k, kind), im in zip(axes.ravel(), picks, imgs):
        ax.imshow(im); ax.axis("off")
        ax.set_title(f"{kind}\nP(tumor)={prob[k]:.2f}", fontsize=9)
    fig.tight_layout(h_pad=3.0); fig.savefig(f"results/misclassified_{out}.png", dpi=130); plt.close(fig)

    # ---- Grad-CAM (optional): 4 correct + 4 incorrect, random (seeded) ----
    if not args.no_gradcam:
        layer = model.m.layer4[-1] if args.model == "resnet18" else model.features[14]
        right = np.flatnonzero(pred == y)
        wrong = np.flatnonzero(pred != y)
        sel = [(k, "correct") for k in rng.choice(right, 4, replace=False)] + \
              [(k, "WRONG") for k in rng.choice(wrong, 4, replace=False)]
        imgs = read_patches(args.root, args.split, [ds.indices[k] for k, _ in sel])
        fig, axes = plt.subplots(2, 8, figsize=(18, 4.8))
        for c, ((k, kind), im) in enumerate(zip(sel, imgs)):
            cam, pr = grad_cam(model, layer, im, device)
            axes[0, c].imshow(im); axes[0, c].axis("off")
            axes[0, c].set_title(f"{kind}: true={'tumor' if y[k] else 'normal'}\nP(tumor)={pr:.2f}", fontsize=8)
            axes[1, c].imshow(im); axes[1, c].imshow(cam, cmap="jet", alpha=0.45); axes[1, c].axis("off")
        fig.tight_layout(); fig.savefig(f"results/gradcam_{out}.png", dpi=130); plt.close(fig)

    print(f"saved metrics + plots for {out} in results/")


if __name__ == "__main__":
    main()
