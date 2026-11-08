# Detecting metastatic tissue in lymph node patches (PatchCamelyon)

**Status: work in progress. The full-data training run and test-set evaluation have not been done yet, so this README contains no performance numbers.**

## The question
Can a convolutional neural network tell whether the center of a small lymph node pathology image contains metastatic (tumor) tissue, and does a pretrained network (transfer learning) do better than a small network trained from scratch?

## The data
[PatchCamelyon (PCam)](https://github.com/basveeling/pcam): 327,680 color patches, 96x96 px, binary label (tumor present in the center region or not). Fixed splits: 262,144 train / 32,768 validation / 32,768 test, used exactly as provided.

**No data is committed to this repo.** To download it (about 7.5 GB):
```
python scripts/download_data.py --root data
```
This uses `torchvision.datasets.PCAM` (Google Drive). Class balance per split is saved to `results/class_balance.csv`, and a grid of sample patches to `results/sample_patches.png`.

## Methods
- **Models:** (a) a 4-block CNN trained from scratch; (b) ResNet18 pretrained on ImageNet, all layers fine-tuned.
- **Augmentation (training only):** random flips, random 90-degree rotations, mild color jitter.
- **Model selection:** the best epoch is chosen by validation AUC only. The test set is scored once per model, at the end, with a fixed 0.5 threshold; `src/evaluate.py` refuses to re-score it.
- **Evaluation:** AUC with a bootstrap 95% confidence interval (2,000 resamples), accuracy, sensitivity, specificity, confusion matrix, ROC curve, calibration plot, misclassified examples, Grad-CAM.
- **Reproducibility:** seeds are fixed (42). GPU runs are not bit-for-bit deterministic, so reruns may differ slightly.
- **Run it:** locally on a subsample (`python -m src.train --model cnn --n-train 20000 --n-val 4000 --epochs 5 --tag dev`) or in full on Colab with `colab_full_run.ipynb`.

## Results
Pending. This section will be filled in from the real output files (`results/metrics.csv`) after the full run.

## Limitations
- These are image **patches, not whole slides**. This is not a diagnostic tool.
- PCam was sampled to be close to **50/50 class balance**, so accuracy does not reflect real-world prevalence of metastasis.
- Each label depends **only on the center region** of the patch.
- The data derive from **400 slides from the Camelyon16 challenge**, so there is no external validation.
- Compute limits meant **no hyperparameter search**.

## How it was built
I used Claude Code for implementation and made the analysis decisions myself.
