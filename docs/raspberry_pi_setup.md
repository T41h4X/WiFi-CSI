# Raspberry Pi Operating Environment

## 1. Operating System Details
The Raspberry Pi 5 runs a modern Debian ARM64 distribution:

* **OS Distribution**: Debian GNU/Linux 13 (Trixie)
* **Architecture**: ARM64 / aarch64
* **Kernel**: `6.18.39+rpt-rpi-v8`
* **Page Size**: 4 KB memory pages

---

## 2. Package Prerequisites
To compile and operate the Nexmon tools, capture packets, and execute the decoders:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    libpcap-dev \
    tcpdump \
    python3 \
    python3-pip \
    python3-venv \
    net-tools \
    wireless-tools
```

---

## 3. Network Interface States
* **Interface**: `wlan0`
* **MAC Address**: Broadcast monitor interface
* **Kernel Driver Module**: `brcmfmac` patched with the Nexmon firmware hook.
* **Verification**:
  ```bash
  uname -a
  dmesg | grep -i brcm
  ip link show wlan0
  ```
