# Project Limitations & Research Boundaries

Honest scientific accounting requires acknowledging current project limitations:

---

## 1. Domain Shift Between Benchmark and Hardware
* **Public Benchmark (UT-HAR)**:
  * 3 receiver antennas $\times$ 30 subcarriers = 90 spatial-frequency channels.
  * Isochronous sampling rate (typically 20 Hz to 100 Hz).
  * High signal-to-noise ratio in a controlled research setting.
* **Raspberry Pi 5 Setup (Nexmon)**:
  * 1 receiver antenna $\times$ 64 subcarriers (with ~54 usable non-guard carriers).
  * Variable packet arrival rate (mean ~16.15 Hz, but bursty with gaps).
* **Limitation**: The CNN models documented here were validated on the UT-HAR benchmark. Transferring weights directly to the Raspberry Pi stream requires domain adaptation, subcarrier realignment, and temporal interpolation.

---

## 2. Irregular Hardware Sampling Intervals
As documented in the physical capture analysis (`kftn_csi.pcap`):
* While the overall average frame rate is 16.15 frames/s, 2,494 intervals exceeded 0.1s and 442 intervals exceeded 1.0s.
* Real-time CNN inference assumes uniform temporal spacing ($\Delta t = \text{const}$). Running real-time inference on the Pi requires an interpolation buffer (e.g. cubic spline or linear interpolation to 20 Hz).

---

## 3. Physical Channel Anomalies
* Subcarriers `[28..35]` on the Broadcom BCM43455 show massive, static amplitude peaks (mean amplitude ~34,916 on SC 29).
* These carriers must be explicitly masked out to prevent saturation of the input tensors.

---

## 4. Class Imbalance
In the 4-class training set:
* Walking accounts for 1,172 samples (51.8% of the dataset).
* Standing accounts for only 243 samples (10.7% of the dataset).
* Although inverse-frequency weighting was applied in the loss function, collecting a more balanced dataset or applying temporal data augmentation would further improve posture recall.
