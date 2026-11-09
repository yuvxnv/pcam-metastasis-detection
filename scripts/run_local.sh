#!/bin/bash
# Local training run. Default: both models on a fixed 60,000-image balanced subsample
# of the training set. Full data:  NTRAIN=all EPOCHS=6 caffeinate -i bash scripts/run_local.sh
# Best epoch is picked on the FULL validation set.
# Does NOT touch the test set. Run from anywhere:  caffeinate -i bash scripts/run_local.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
EPOCHS=${EPOCHS:-8}
NTRAIN=${NTRAIN:-60000}   # a number, or "all" for the full 262,144 training images
if [ "$NTRAIN" = "all" ]; then NARG=""; else NARG="--n-train $NTRAIN"; fi

for m in cnn resnet18; do
  echo "=== $(date '+%H:%M:%S') training $m ===" | tee -a results/run_local.log
  $PY -m src.train --model $m $NARG --epochs $EPOCHS 2>&1 | grep --line-buffered -vi warning | tee -a results/run_local.log
done

echo "=== $(date '+%H:%M:%S') validation evaluation ===" | tee -a results/run_local.log
for m in cnn resnet18; do
  $PY -m src.evaluate --model $m --split val 2>&1 | grep --line-buffered -vi warning | tee -a results/run_local.log
done
echo "=== $(date '+%H:%M:%S') ALL DONE (test set NOT touched). Show results/run_local.log to Claude. ===" | tee -a results/run_local.log
