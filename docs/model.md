# Model Architecture: CSI_CNN_V2

## 1. Why Convolutional Neural Networks for Wi-Fi CSI?
Channel State Information (CSI) can be viewed as a 2D matrix spanning **Time** along one axis and **Subcarrier Frequency** along the other:
* **Time Axis**: Reflects human movement cadence, velocity, and duration.
* **Frequency / Subcarrier Axis**: Reflects frequency-selective fading and spatial multipath characteristics.

A 2D Convolutional Neural Network (CNN) is uniquely suited for CSI because:
1. **Local Connectivity**: 2D kernels ($3 \times 3$) detect localized spatio-temporal features—such as periodic crests and troughs caused by walking strides or transient phase shifts caused by sitting down.
2. **Translation Invariance**: The model recognizes motion patterns even if an action starts earlier or later within the 250-sample window.
3. **Parameter Efficiency**: Compared to large Fully Connected MLPs ($250 \times 90 = 22,500$ input features per sample), weight sharing drastically limits parameter count to only 93,636 weights, preventing overfitting on small-to-medium datasets.

---

## 2. Detailed Layer Stack & Tensor Shapes

| Layer / Block | Operation | Output Shape `(B, C, H, W)` | Parameter Count |
| :--- | :--- | :--- | :--- |
| **Input** | Raw standardized tensor | `(B, 1, 250, 90)` | 0 |
| **Block 1** | `Conv2d(1, 32, kernel=3, pad=1)` | `(B, 32, 250, 90)` | $32 \times (1 \times 3 \times 3) + 32 = 320$ |
| | `BatchNorm2d(32)` | `(B, 32, 250, 90)` | $32 \times 2 = 64$ |
| | `ReLU(inplace=True)` | `(B, 32, 250, 90)` | 0 |
| | `MaxPool2d(2)` | `(B, 32, 125, 45)` | 0 |
| **Block 2** | `Conv2d(32, 64, kernel=3, pad=1)` | `(B, 64, 125, 45)` | $64 \times (32 \times 3 \times 3) + 64 = 18,496$ |
| | `BatchNorm2d(64)` | `(B, 64, 125, 45)` | $64 \times 2 = 128$ |
| | `ReLU(inplace=True)` | `(B, 64, 125, 45)` | 0 |
| | `MaxPool2d(2)` | `(B, 64, 62, 22)` | 0 |
| **Block 3** | `Conv2d(64, 128, kernel=3, pad=1)`| `(B, 128, 62, 22)` | $128 \times (64 \times 3 \times 3) + 128 = 73,856$ |
| | `BatchNorm2d(128)` | `(B, 128, 62, 22)` | $128 \times 2 = 256$ |
| | `ReLU(inplace=True)` | `(B, 128, 62, 22)` | 0 |
| | `MaxPool2d(2)` | `(B, 128, 31, 11)` | 0 |
| **Classifier** | `AdaptiveAvgPool2d((1, 1))` | `(B, 128, 1, 1)` | 0 |
| | `Flatten()` | `(B, 128)` | 0 |
| | `Dropout(p=0.30)` | `(B, 128)` | 0 |
| | `Linear(128, 4)` | `(B, 4)` | $4 \times 128 + 4 = 516$ |
| **Total** | | | **93,636 parameters** |

All 93,636 parameters are trainable.
