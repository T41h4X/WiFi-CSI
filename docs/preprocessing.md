# Data Preprocessing & Normalization

## 1. Class Filtering & Relabeling
Executed by `training/prepare_4class_dataset.py`:
1. Loads original `X_train.csv`, `X_val.csv`, `X_test.csv` (which are binary numpy arrays formatted with .csv filenames).
2. Applies a Boolean mask to select only target label integers $\{0, 2, 5, 6\}$.
3. Maps selected labels monotonically to $\{0, 1, 2, 3\}$.
4. Saves pre-filtered splits as clean `.npy` files in `Data/UT_HAR_4CLASS/`:
   * `X_train.npy` (2260, 250, 90, float32)
   * `y_train.npy` (2260, int64)
   * `X_val.npy` (282, 250, 90, float32)
   * `y_val.npy` (282, int64)
   * `X_test.npy` (284, 250, 90, float32)
   * `y_test.npy` (284, int64)

---

## 2. Normalization Strategy: Baseline vs Optimized

### Baseline (v1: `csi_dataset.py`)
* Applied **no normalization** at all.
* Raw amplitude values were fed directly into the model.
* Result: Gradients were unbalanced across dynamic and static activities, leading to lower test accuracy (86.62%).

### Optimized (v2: `csi_dataset_v2.py`)
* Implements **Z-score standardization** strictly derived from the training set to prevent data leakage:
  $$\mu = \frac{1}{N_{\text{train}}} \sum X_{\text{train}}, \quad \sigma = \sqrt{\frac{1}{N_{\text{train}}} \sum (X_{\text{train}} - \mu)^2}$$
* Training split statistics:
  * $\mu \approx 0.082776$
  * $\sigma \approx 0.054368$
* All splits (train, validation, test) are transformed using these exact training parameters:
  $$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$
* Result: Inputs are zero-centered with unit variance, stabilizing batch normalization layers and accelerating convergence.

---

## 3. Class Imbalance Weighting
Because walking represents 51.8% of the dataset while standing accounts for only 10.7%, unweighted training biases the classifier toward the majority class. Inverse-frequency class weighting is computed:
$$w_c = \frac{N_{\text{total}}}{C \cdot N_c}$$
For the 4 classes:
* Lying ($N=525$): $w_0 = 1.0762$
* Walking ($N=1172$): $w_1 = 0.4821$
* Sitting ($N=320$): $w_2 = 1.7656$
* Standing ($N=243$): $w_3 = 2.3251$
These weights are passed to `nn.CrossEntropyLoss(weight=class_weights)`.
