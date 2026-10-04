# Training Pipeline & Model Definitions

This directory contains the dataset preparation scripts, PyTorch Dataset definitions, and training workflows for CSI-based Human Activity Recognition.

---

## 1. Directory Structure

```
training/
├── prepare_4class_dataset.py  # Filters 4 target classes from UT-HAR and standardizes labels
├── inspect_ut_har.py          # Visualizer generating activity-specific CSI spectrograms
├── csi_dataset.py             # PyTorch Dataset implementation for baseline v1 (unnormalized)
├── csi_dataset_v2.py          # PyTorch Dataset v2 (Train-set z-score standardization)
├── train_cnn.py               # Baseline CNN training script (v1)
├── train_cnn_v2.py            # Optimized CNN training script (v2, lr scheduler, early stopping)
├── train_cnn_v2_multiseed.py  # Multi-seed validation harness (Seeds: 42, 123, 456, 789, 2026)
└── README.md
```

---

## 2. Target Classes & Dataset Preparation

Run `prepare_4class_dataset.py` to extract and remap the four core activities from the raw UT-HAR benchmark files:
* **Original 0 ("lie down")** $\rightarrow$ **0: Lying** (525 train / 66 val / 66 test)
* **Original 2 ("walk")** $\rightarrow$ **1: Walking** (1172 train / 146 val / 147 test)
* **Original 5 ("sit down")** $\rightarrow$ **2: Sitting** (320 train / 40 val / 40 test)
* **Original 6 ("stand up")** $\rightarrow$ **3: Standing** (243 train / 30 val / 31 test)

Total: 2,826 samples (2,260 train, 282 validation, 284 test).

---

## 3. Training Commands

### Baseline Model (v1)
```bash
python training/train_cnn.py
```

### Optimized Model (v2)
```bash
python training/train_cnn_v2.py
```

### Multi-Seed Statistical Validation
```bash
python training/train_cnn_v2_multiseed.py
```
Outputs validation statistics across 5 seeds directly to `results/multiseed/csi_har_cnn_v2_multiseed_results.csv`.
