#!/usr/bin/env bash
# ==============================================================================
# Start Raw CSI Packet Capture on Port 5500
# Target: BSSID 58:D9:D5:28:7D:B1 on Channel 6 (20 MHz)
# ==============================================================================

set -euo pipefail

OUTPUT_PCAP="${1:-kftn_csi.pcap}"
PORT=5500
IFACE="wlan0"

echo "=========================================================="
echo "          Starting Wi-Fi CSI Capture Service              "
echo "=========================================================="
echo "[*] Interface       : ${IFACE}"
echo "[*] Target UDP Port : ${PORT}"
echo "[*] Output PCAP     : ${OUTPUT_PCAP}"
echo "[*] Press Ctrl+C to terminate capture."
echo "=========================================================="

# Run tcpdump capturing CSI extraction UDP datagrams
sudo tcpdump -i "${IFACE}" -vv -s 0 "dst port ${PORT}" -w "${OUTPUT_PCAP}"
