# Research Results & Evaluation Artifacts

This directory contains empirical evaluation metrics, confusion matrices, training progression curves, and qualitative CSI visualizations.

---

## 1. Directory Structure

```
results/
├── confusion_matrix/
│   ├── csi_har_cnn_v2_confusion_matrix.npy  # 4x4 NumPy integer matrix
│   └── csi_har_cnn_v2_confusion_matrix.png  # Heatmap visualization
├── multiseed/
│   └── csi_har_cnn_v2_multiseed_results.csv # Metrics across 5 seeds
├── plots/
│   ├── csi_har_cnn_v2_loss_curve.png        # Training vs validation loss
│   ├── csi_har_cnn_v2_accuracy_curve.png    # Training vs validation accuracy
│   ├── csi_har_cnn_v2_f1_curve.png          # Training vs validation macro-F1
│   ├── example_lying.png                    # CSI amplitude spectrogram for Lying
│   ├── example_sitting.png                  # CSI amplitude spectrogram for Sitting
│   ├── example_standing.png                 # CSI amplitude spectrogram for Standing
│   ├── example_walking.png                  # CSI amplitude spectrogram for Walking
│   └── four_target_activities.png           # 2x2 comparison grid of all 4 classes
├── test_confusion_matrix.npy                # Baseline v1 confusion matrix
├── test_labels.npy                          # Baseline v1 ground truth test labels
├── test_predictions.npy                     # Baseline v1 predicted test labels
└── README.md
```

---

## 2. Quantitative Summary

### Model Version Comparison (Test Set, N = 284)
| Metric | Baseline CNN (v1) | Optimized CNN (v2, Seed 42) | Multi-Seed v2 Mean ± Std (5 Seeds) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 86.62% | **94.37%** | **91.41% ± 2.42%** |
| **Macro Precision** | 81.33% | **90.14%** | **86.14% ± 3.64%** |
| **Macro Recall** | 83.21% | **91.70%** | **86.81% ± 3.56%** |
| **Macro F1** | 85.55% | **90.66%** | **85.86% ± 3.70%** |

### Confusion Matrix Breakdown (v2, Seed 42)
* **Lying**: 58 / 66 correct (87.88% recall). 5 misclassified as Sitting, 3 as Standing.
* **Walking**: 146 / 147 correct (99.32% recall). 1 misclassified as Sitting.
* **Sitting**: 37 / 40 correct (92.50% recall). 3 misclassified as Standing.
* **Standing**: 27 / 31 correct (87.10% recall). 4 misclassified as Sitting.
