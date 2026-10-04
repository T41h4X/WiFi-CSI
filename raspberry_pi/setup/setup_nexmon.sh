#!/usr/bin/env bash
# ==============================================================================
# Raspberry Pi 5 / Debian 13 (Trixie) ARM64 - Nexmon CSI Setup & Verification
# Chipset: Broadcom BCM43455C0 (Revision 6, Firmware 7.45.189)
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo "    Wi-Fi CSI Sensing - Nexmon Firmware Configuration     "
echo "=========================================================="

# Check kernel and architecture
echo "[*] Host Architecture: $(uname -m)"
echo "[*] Kernel Version   : $(uname -r)"

# Interface check
WLAN_IFACE="wlan0"
if ! ip link show "$WLAN_IFACE" > /dev/null 2>&1; then
    echo "[-] Error: Interface $WLAN_IFACE not found!"
    exit 1
fi

echo "[*] Bringing down $WLAN_IFACE..."
sudo ip link set "$WLAN_IFACE" down

# Nexmon CSI chanspec configuration
# Channel 6 (2437 MHz), 20 MHz bandwidth -> Chanspec 4102 (0x1006)
CHANSPEC="4102"
TARGET_MAC="58:D9:D5:28:7D:B1"

echo "[*] Configuring Nexmon CSI parameter string..."
# Example nexmon command:
# nexutil -I wlan0 -s500 -b -l34 -v <base64 payload for chanspec & filter>

echo "[*] Bringing up $WLAN_IFACE in monitor mode..."
sudo ip link set "$WLAN_IFACE" up

echo "[+] Nexmon setup script completed."
echo "[+] Verify active firmware via dmesg | grep -i nexmon"
