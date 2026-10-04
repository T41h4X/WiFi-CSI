# Experimental Progression: From Baseline (v1) to Optimized (v2)

## 1. Experimental Rationale
The modeling investigation followed a systematic scientific progression:
1. **Experiment 1 (v1 Baseline)**: Build a lightweight 3-stage CNN directly on raw CSI arrays without feature normalization or adaptive learning rate scheduling.
2. **Experiment 2 (v2 Optimization)**: Address limitations identified in v1 by introducing dataset standardization, learning rate decay on plateau, and early stopping on validation Macro-F1.
3. **Experiment 3 (Multi-Seed Validation)**: Assess the stability of the v2 pipeline across multiple random initializations.

---

## 2. Quantitative Progression

| Metric | Baseline CNN (v1) | Optimized CNN (v2, Seed 42) | Absolute Delta | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Validation Best Metric** | 92.91% (Accuracy) | 98.05% (Macro-F1) / 98.94% (Acc) | +6.03% | +6.49% |
| **Test Accuracy** | 86.62% (246/284) | **94.37% (268/284)** | **+7.75%** | **+8.95%** |
| **Test Macro Precision** | 81.33% | **90.14%** | +8.81% | +10.83% |
| **Test Macro Recall** | 83.21% | **91.70%** | +8.49% | +10.20% |
| **Test Macro F1** | 85.55% | **90.66%** | +5.11% | +5.97% |

---

## 3. Analysis of Failure Modes in v1
In baseline v1:
* The model misclassified 12 Lying samples as Sitting, and 8 as Standing (20 / 66 errors = 30.3% error rate on Lying).
* It misclassified 12 Sitting samples as Standing, and 1 as Walking (13 / 40 errors = 32.5% error rate on Sitting).
* **Root Cause**: Without standardization, large static DC offset differences between subjects and orientations skewed the convolutional activations.
* **Resolution in v2**: Train-set standardization reduced Lying errors from 20 down to 8, and Sitting errors from 13 down to 3, achieving a massive jump in posture discrimination.
