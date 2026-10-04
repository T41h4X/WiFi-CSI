# Reproducibility Guide & Execution Runbook

This runbook guides researchers in replicating both the machine learning experiments and the Raspberry Pi acquisition pipeline.

---

## 1. Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/T41h4X/WiFi-CSI.git
cd WiFi-CSI

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 2. Machine Learning Pipeline Reproduction

### Step 1: Prepare the 4-Class Dataset
Filters classes 0, 2, 5, 6 from the raw UT-HAR data and remaps them to 0, 1, 2, 3:
```bash
python training/prepare_4class_dataset.py
```

### Step 2: Inspect Activity Patterns & Generate Plots
Produces visual spectrograms of each activity in `inspection_results/`:
```bash
python training/inspect_ut_har.py
```

### Step 3: Train the Baseline CNN (v1)
Runs the unnormalized baseline model:
```bash
python training/train_cnn.py
```

### Step 4: Train the Optimized CNN (v2)
Runs the standardized model with early stopping and learning rate scheduling:
```bash
python training/train_cnn_v2.py
```

### Step 5: Execute Multi-Seed Statistical Validation
Runs 5 independent runs (seeds 42, 123, 456, 789, 2026) and updates `results/multiseed/csi_har_cnn_v2_multiseed_results.csv`:
```bash
python training/train_cnn_v2_multiseed.py
```

---

## 3. Raspberry Pi Acquisition Reproduction

On the Raspberry Pi 5 with patched Nexmon CSI firmware:

### Step 1: Configure Monitor Interface
```bash
chmod +x raspberry_pi/setup/setup_nexmon.sh
sudo ./raspberry_pi/setup/setup_nexmon.sh
```

### Step 2: Start Packet Capture
```bash
chmod +x raspberry_pi/capture/start_capture.sh
./raspberry_pi/capture/start_capture.sh kftn_csi.pcap
```

### Step 3: Decode Raw PCAP to CSV
```bash
python raspberry_pi/decoder/decode_nexmon_csi.py
```

### Step 4: Run Statistical & Temporal Verification
```bash
python raspberry_pi/analysis/analyze_csi.py
python raspberry_pi/analysis/csi_temporal_check.py
python raspberry_pi/analysis/plot_csi.py
```
Outputs amplitude time-series, subcarrier distributions, and 2D heatmaps.
