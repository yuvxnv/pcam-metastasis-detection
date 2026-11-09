"""Pick a decision threshold using the VALIDATION set only.

Rule (fixed in advance): the highest threshold whose validation sensitivity is >= TARGET.
That keeps specificity as high as possible while still catching at least 90% of tumor patches.
The chosen number is saved and later applied, unchanged, to the test set.

  python -m src.choose_threshold --model resnet18
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

TARGET_SENSITIVITY = 0.90


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "resnet18"], required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--target", type=float, default=TARGET_SENSITIVITY)
    args = p.parse_args()
    name = args.model + (f"_{args.tag}" if args.tag else "")

    d = pd.read_csv(f"results/preds_{name}_val.csv")
    if len(d) != 32768:
        raise SystemExit(f"Need predictions for the FULL validation set (got {len(d)}).")
    y, prob = d["label"].to_numpy(), d["prob"].to_numpy()

    fpr, tpr, thr = roc_curve(y, prob)           # thresholds are in decreasing order
    i = int(np.argmax(tpr >= args.target))       # first (= highest) threshold reaching the target
    threshold = float(thr[i])
    out = {"model": name, "rule": f"highest threshold with validation sensitivity >= {args.target}",
           "threshold": threshold, "val_sensitivity": float(tpr[i]), "val_specificity": float(1 - fpr[i]),
           "chosen_on": "validation set only"}
    json.dump(out, open(f"results/threshold_{name}.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
