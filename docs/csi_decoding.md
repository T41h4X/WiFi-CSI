# CSI Packet Structure & Python Decoder

## 1. Nexmon UDP Packet Structure
Nexmon wraps raw channel measurements in a custom UDP payload sent to port `5500`.

### Header Layout (18 Bytes)
| Offset (Bytes) | Field Name | Type | Description |
| :--- | :--- | :--- | :--- |
| `0x00 - 0x01` | Magic | `uint16_t` (Little-endian) | Always `0x1111` |
| `0x02 - 0x03` | RSSI | `int16_t` | Frame RSSI |
| `0x04 - 0x09` | Source MAC | `uint8_t[6]` | MAC address of transmitter |
| `0x0A - 0x0B` | Sequence | `uint16_t` | Frame sequence counter |
| `0x0C - 0x0D` | Core | `uint16_t` | Radio core index (e.g. 0) |
| `0x0E - 0x0F` | Chanspec | `uint16_t` | Channel/bandwidth code (`4102`) |
| `0x10 - 0x11` | Chip ID | `uint16_t` | Chip identifier (`101` / `0x65`) |

### CSI Payload Layout (256 Bytes)
Following the 18-byte header are $N_{\text{FFT}} = 64$ complex samples:
* Each sample is 4 bytes: 2 bytes signed 16-bit integer for the Real component, 2 bytes signed 16-bit integer for the Imaginary component.
* Total CSI payload length: $64 \times 4 = 256$ bytes.
* Total UDP payload length: $18 + 256 = 274$ bytes.

---

## 2. Mathematical Transformations
For each subcarrier $i \in [0, 63]$:
$$\text{Amplitude}_i = \sqrt{\text{Real}_i^2 + \text{Imag}_i^2}$$
$$\text{Phase}_i = \operatorname{atan2}(\text{Imag}_i, \text{Real}_i)$$

---

## 3. Decoder Implementation (`decode_nexmon_csi.py`)
The Python decoder reads raw `.pcap` files directly without external packet dependencies:
1. Validates the PCAP global header (magic `0xa1b2c3d4` or `0xd4c3b2a1`).
2. Iterates over packet records, extracting Ethernet $\rightarrow$ IPv4 $\rightarrow$ UDP layers.
3. Filters for UDP destination port `5500` and magic `0x1111`.
4. Decodes 64 complex subcarrier numbers via `struct.unpack_from('<hh', csi_data, offset)`.
5. Computes amplitude and phase.
6. Writes a structured CSV with 262 columns:
   ```python
   fieldnames = [
       "timestamp", "src_mac", "sequence", "core", "chanspec", "chip",
       *[f"sc_{i}_real" for i in range(64)],
       *[f"sc_{i}_imag" for i in range(64)],
       *[f"sc_{i}_amplitude" for i in range(64)],
       *[f"sc_{i}_phase" for i in range(64)]
   ]
   ```
