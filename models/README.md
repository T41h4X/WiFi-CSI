# Trained Model Checkpoints

This directory houses the PyTorch model checkpoints trained on the 4-class Wi-Fi CSI Human Activity Recognition benchmark.

---

## 1. Checkpoint Inventory

| Checkpoint File | Size | Architecture | Training Config | Val Macro-F1 / Acc | Test Accuracy | Test Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`csi_har_cnn_best.pth`** | 384 KB | `CSI_CNN` (v1) | 30 Epochs, No normalization, unseeded | 92.91% (Acc) | 86.62% | 85.55% |
| **`csi_har_cnn_v2_best.pth`** | 384 KB | `CSI_CNN_V2` (v2) | 50 Epochs, Train z-score, LR scheduler, Early stopping, Seed 42 | **98.05% (F1) / 98.94% (Acc)** | **94.37%** | **90.66%** |

---

## 2. Checkpoint Details

### `csi_har_cnn_v2_best.pth` (Primary Production Weights)
Saved from epoch 46 of `train_cnn_v2.py`.
Contains the following dictionary keys:
* `model_state_dict`: Weight tensors for all 3 Conv2d blocks, 3 BatchNorm layers, and 1 Linear classifier.
* `epoch`: 46
* `val_f1`: `0.980499` (Validation Macro-F1)
* `val_accuracy`: `0.989362` (Validation Accuracy: 98.94%)
* `seed`: 42

### Model Loading Example
```python
import torch
from training.train_cnn_v2 import CSI_CNN_V2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSI_CNN_V2(num_classes=4).to(device)

checkpoint = torch.load("models/csi_har_cnn_v2_best.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded CSI_CNN_V2 successfully from epoch {checkpoint['epoch']}!")
```
