# Datasets & Data Acquisition Guidelines

This project utilizes two primary tiers of Channel State Information (CSI) data:
1. **Public Research Benchmark (UT-HAR)**: Used for training and validating the deep neural networks.
2. **Real-World Raspberry Pi Nexmon CSI**: Collected directly from a Raspberry Pi 5 with an integrated Broadcom BCM43455C0 Wi-Fi chipset.

---

## 1. Public Benchmark: UT-HAR 4-Class Dataset

* **Source**: Yousefi et al., *"A Survey on Behavior Recognition Using WiFi Channel State Information"*, IEEE Communications Magazine, 2017.
* **Format**: NumPy `.npy` arrays shaped `(N, 250, 90)` corresponding to 250 time steps and 90 subcarrier channels (3 receiver antennas × 30 subcarriers).
* **Target Classes**:
  * `0`: Lying (525 train / 66 val / 66 test)
  * `1`: Walking (1172 train / 146 val / 147 test)
  * `2`: Sitting (320 train / 40 val / 40 test)
  * `3`: Standing (243 train / 30 val / 31 test)
* **Storage Policy**: Because full benchmark matrices exceed GitHub's standard upload limits (`X_train.csv` is 682 MB and `X_train.npy` is 194 MB), they are stored locally and excluded from git commits via `.gitignore`.
* **Reproduction**:
  ```bash
  python training/prepare_4class_dataset.py
  ```

---

## 2. Raspberry Pi Nexmon CSI Dataset (`kftn_csi.pcap` / `kftn_csi.csv`)

* **Capture File**: `kftn_csi.pcap` (9.44 MB)
* **Decoded Output**: `kftn_csi.csv` (84.7 MB, 28,425 rows × 262 columns)
* **Access Point**: SSID `KFTN`, BSSID `58:D9:D5:28:7D:B1`, Channel 6 (20 MHz, chanspec 4102)
* **Structure**:
  * 6 metadata fields: `timestamp`, `src_mac`, `sequence`, `core`, `chanspec`, `chip`
  * 64 real values (`sc_0_real` to `sc_63_real`)
  * 64 imaginary values (`sc_0_imag` to `sc_63_imag`)
  * 64 amplitude values (`sc_0_amplitude` to `sc_63_amplitude`)
  * 64 phase values (`sc_0_phase` to `sc_63_phase`)
* **Decoder Execution**:
  ```bash
  python raspberry_pi/decoder/decode_nexmon_csi.py
  ```
