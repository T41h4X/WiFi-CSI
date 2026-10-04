# Hardware Setup & RF Environment

## 1. Physical Equipment
The physical sensing testbed comprises commercial off-the-shelf (COTS) devices without proprietary radar hardware:

| Component | Specification | Role in System |
| :--- | :--- | :--- |
| **SBC** | Raspberry Pi 5 Model B (8 GB RAM) | Dedicated receiver, capture platform, and edge decoder |
| **Wi-Fi Chipset** | Broadcom / Cypress BCM43455C0 (Revision 6) | 1×1 SISO 802.11ac physical-layer transceiver |
| **Transmitter** | COTS Wi-Fi Access Point | Transmits ambient 802.11n beacons and data frames |
| **Camera (Optional)** | Sony IMX219 (Raspberry Pi Camera Module v2) | Auxiliary ground-truth video logger (part of FYDP setup) |

---

## 2. RF Network Configuration
To maintain controlled physical-layer subcarrier tracking, the receiver and access point operate on dedicated RF parameters:

* **SSID**: `KFTN`
* **Transmitter BSSID / MAC**: `58:D9:D5:28:7D:B1`
* **Operating Frequency Band**: 2.4 GHz ISM
* **Primary Wi-Fi Channel**: Channel 6 (Center Frequency: 2437 MHz)
* **Channel Bandwidth**: 20 MHz
* **Nexmon Chanspec ID**: `4102` (`0x1006`)
* **Subcarrier FFT Size**: 64 subcarriers (spacing: 312.5 kHz)

---

## 3. Placement & Multipath Dynamics
* **Geometry**: The AP and Raspberry Pi 5 are positioned in a typical indoor room (approximately 1.0 to 1.2 meters above ground level, separated by 3 to 5 meters).
* **Line of Sight (LoS)**: The direct path between transmitter and receiver provides dominant static carrier energy.
* **Non-Line of Sight (NLoS)**: Dynamic reflection paths off human limbs introduce phase delay and constructive/destructive interference, enabling the sensing of torso motion, walking cadence, and postural shifts.
