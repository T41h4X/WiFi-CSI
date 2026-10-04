# Raspberry Pi 5 CSI Acquisition & Preprocessing Pipeline

This directory contains the operational and decoding suite deployed on the **Raspberry Pi 5 (8 GB)** for extracting raw Wi-Fi Channel State Information (CSI) from 802.11n Wi-Fi frames using **Nexmon CSI**.

---

## 1. Hardware & System Context

* **SBC**: Raspberry Pi 5 (8 GB RAM)
* **Wi-Fi Chipset**: Broadcom BCM43455C0 (Revision 6)
* **Operating System**: Debian GNU/Linux 13 (Trixie) ARM64 / aarch64
* **Kernel Version**: `6.18.39+rpt-rpi-v8` (4 KB memory pages)
* **Nexmon CSI Build**: Firmware `7.45.189`, Chip BCM43455C0 rev 6, build `a975-dirty-3`
* **Transmitter Access Point**:
  * **SSID**: `KFTN`
  * **BSSID**: `58:D9:D5:28:7D:B1`
  * **Channel**: Channel 6 (2.437 GHz, 20 MHz bandwidth, chanspec `4102`)

---

## 2. Directory Architecture

```
raspberry_pi/
├── setup/
│   └── setup_nexmon.sh       # Interface initialization and chanspec configuration
├── capture/
│   └── start_capture.sh      # tcpdump listener on UDP port 5500
├── decoder/
│   └── decode_nexmon_csi.py  # Binary PCAP parser extracting 64 subcarriers to CSV
├── analysis/
│   ├── analyze_csi.py        # Statistical validator (frame counts, interval stats)
│   ├── csi_temporal_check.py # Per-subcarrier variance and temporal stability check
│   └── plot_csi.py           # Amplitude time-series, heatmap, and distribution generator
└── README.md
```

---

## 3. The Extraction & Decoding Process

1. **Nexmon CSI Driver**: The patched firmware intercepts physical-layer 802.11 frames from target MAC `58:D9:D5:28:7D:B1`, extracts the Channel State Information matrix (64 subcarriers), attaches a Nexmon CSI header (`0x1111` magic), and transmits it as an internal UDP payload to port `5500`.
2. **PCAP Capture**: `start_capture.sh` writes the UDP packets directly to `kftn_csi.pcap`.
3. **Payload Structure**:
   * **Nexmon Header (18 bytes)**: Magic (`0x1111`), source MAC address, sequence number, core ID, chanspec, chip ID.
   * **CSI Data (256 bytes)**: 64 complex subcarrier samples. Each sample contains a 16-bit signed real component (`int16`) and a 16-bit signed imaginary component (`int16`).
   $$\text{Amplitude} = \sqrt{\text{Real}^2 + \text{Imag}^2}, \quad \text{Phase} = \operatorname{atan2}(\text{Imag}, \text{Real})$$
4. **CSV Export**: `decode_nexmon_csi.py` parses each packet and writes a 262-column CSV (`timestamp`, 6 metadata fields, 64 real, 64 imag, 64 amplitude, 64 phase).

---

## 4. Empirical Signal Characteristics

* **Capture Duration**: 1,759.72 seconds (~29.33 minutes)
* **Frames Captured**: 28,425 frames (0 kernel drops)
* **Nominal Mean Frame Rate**: ~16.15 frames/sec
* **Median Interval**: ~0.0025 sec (burst delivery)
* **Guard / Null Subcarriers**: Subcarriers `[0, 28, 29, 30, 31, 32, 33, 34, 35]` correspond to DC/null and central pilot carriers with static, anomalous amplitude values (e.g. Subcarrier 29 mean amplitude ≈ 34,916). These are masked out during sensing visualization and modeling.
