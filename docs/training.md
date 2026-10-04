# Training Configuration & Protocols

## 1. Hyperparameters

| Configuration Parameter | Baseline (v1: `train_cnn.py`) | Optimized (v2: `train_cnn_v2.py`) |
| :--- | :--- | :--- |
| **Input Shape** | `(B, 1, 250, 90)` | `(B, 1, 250, 90)` |
| **Batch Size** | 32 | 32 |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adam ($\beta_1=0.9, \beta_2=0.999$) |
| **Initial Learning Rate** | `0.001` | `0.001` |
| **Weight Decay** | `1e-4` | `1e-4` |
| **Max Epochs** | 30 | 50 |
| **Early Stopping** | None (runs full 30 epochs) | `patience = 8` epochs without validation F1 gain |
| **Learning Rate Scheduler** | None (constant LR) | `ReduceLROnPlateau(mode="max", factor=0.5, patience=3)` |
| **Loss Function** | Weighted Cross-Entropy | Weighted Cross-Entropy (`[1.0762, 0.4821, 1.7656, 2.3251]`) |
| **Model Selection Metric** | Validation Accuracy | Validation Macro-F1 |
| **Random Seed** | Unspecified | Seed 42 (and 5 seeds in multi-seed validation) |

---

## 2. Dynamic Learning Rate Scheduling
In v2, the learning rate adapts automatically to validation progress:
* Monitored Metric: Validation Macro-F1.
* Decay Rule: If validation F1 fails to improve for 3 consecutive epochs, the learning rate is halved ($\text{LR} \leftarrow 0.5 \times \text{LR}$).
* Effect: Enables rapid coarse convergence in initial epochs, followed by fine-grained optimization as the model approaches local minima.

---

## 3. Early Stopping Regularization
To prevent overfitting:
* The training loop caches the best model weights whenever validation Macro-F1 achieves a new peak.
* A counter tracks `epochs_without_improvement`.
* If no improvement is observed for 8 epochs, training terminates early, and the best model weights are restored for test set evaluation.
