# Nexmon CSI Firmware Architecture & Setup

## 1. Overview
Nexmon is a C-based firmware patching framework for Broadcom/Cypress Wi-Fi chips. Standard Wi-Fi drivers discard the raw channel frequency response (CFR) after equalization and frame reception; Nexmon hooks into the firmware's internal frame reception routines to extract the CSI matrix before it is discarded, encapsulates it, and routes it to the host network stack.

---

## 2. Firmware Version & Build Metadata
* **Chip Model**: BCM43455C0
* **Chip Revision**: 6
* **Firmware Version**: `7.45.189`
* **Nexmon CSI Build Signature**: `a975-dirty-3`
* **Chip Identifier**: `101` (`0x65`)

---

## 3. Configuration & Chanspec Mapping
Nexmon CSI uses a compact chanspec structure to specify channel, bandwidth, and subcarrier format:

* **Channel 6 (20 MHz bandwidth)** $\rightarrow$ chanspec code `4102` (`0x1006`).
* The firmware is configured to filter packets originating strictly from transmitter BSSID `58:D9:D5:28:7D:B1`.
* Filtered packets generate a synthetic local UDP datagram transmitted to destination port `5500`.
* Verification command:
  ```bash
  # Check active firmware version via driver ioctl / debugfs
  dmesg | grep -i nexmon
  ```
