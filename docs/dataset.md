# Benchmark Dataset: UT-HAR 4-Class Specification

## 1. Origin & Background
To develop and evaluate a human activity recognition neural network, this project leverages the **UT-HAR** dataset (Yousefi et al., 2017) accessible through the open-source **SenseFi** benchmark (Yang et al., Patterns, Cell Press, 2023).

---

## 2. Original UT-HAR vs Filtered 4-Class Dataset

### Original UT-HAR Classes (7 Classes, 4,973 samples)
| Class ID | Label Name | Train Samples | Val Samples | Test Samples | Total |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | Lie Down | 525 | 66 | 66 | 657 |
| 1 | Fall | 354 | 44 | 45 | 443 |
| 2 | Walk | 1,172 | 146 | 147 | 1,465 |
| 3 | Pickup | 396 | 49 | 50 | 495 |
| 4 | Run | 967 | 121 | 121 | 1,209 |
| 5 | Sit Down | 320 | 40 | 40 | 400 |
| 6 | Stand Up | 243 | 30 | 31 | 304 |
| **Total** | | **3,977** | **496** | **500** | **4,973** |

---

### Selected 4-Class Target Activities (2,826 samples)
To establish a robust baseline focusing on foundational daily human postures and locomotion, four core activities were extracted:

| Old Label | New Label | Activity Name | Train Samples | Val Samples | Test Samples | Total |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | **0** | **Lying** | 525 | 66 | 66 | 657 |
| 2 | **1** | **Walking** | 1,172 | 146 | 147 | 1,465 |
| 5 | **2** | **Sitting** | 320 | 40 | 40 | 400 |
| 6 | **3** | **Standing** | 243 | 30 | 31 | 304 |
| **Total** | | | **2,260** (80.0%) | **282** (10.0%) | **284** (10.0%) | **2,826** |

---

## 3. Tensor Dimensions
* **Raw Array Dimensions**: `(N, 250, 90)`
* **Expanded CNN Input Dimensions**: `(N, 1, 250, 90)`
  * `N`: Number of samples in batch
  * `1`: Single channel (CSI amplitude spectrogram)
  * `250`: Time steps (temporal sampling duration)
  * `90`: Frequency subcarriers (3 receiver antennas × 30 subcarriers per antenna)
