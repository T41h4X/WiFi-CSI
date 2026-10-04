import struct
import csv
import sys
import os
import math


PCAP_FILE = os.path.expanduser("~/kftn_csi.pcap")
OUTPUT_FILE = os.path.expanduser("~/kftn_csi.csv")

NFFT = 64
CSI_HEADER_SIZE = 18


def read_pcap_packets(filename):
    with open(filename, "rb") as f:

        # PCAP global header
        global_header = f.read(24)

        if len(global_header) != 24:
            raise RuntimeError("Invalid PCAP file")

        magic = global_header[:4]

        if magic == b"\xd4\xc3\xb2\xa1":
            endian = "<"
        elif magic == b"\xa1\xb2\xc3\xd4":
            endian = ">"
        else:
            raise RuntimeError(
                f"Unsupported PCAP magic: {magic.hex()}"
            )

        packet_header_format = endian + "IIII"

        while True:
            packet_header = f.read(16)

            if len(packet_header) == 0:
                break

            if len(packet_header) != 16:
                break

            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(
                packet_header_format,
                packet_header
            )

            packet = f.read(incl_len)

            if len(packet) != incl_len:
                break

            yield ts_sec, ts_usec, packet


def extract_udp_payload(packet):
    # Ethernet header
    if len(packet) < 14:
        return None

    ethertype = struct.unpack("!H", packet[12:14])[0]

    # IPv4
    if ethertype != 0x0800:
        return None

    ip = packet[14:]

    if len(ip) < 20:
        return None

    version = ip[0] >> 4

    if version != 4:
        return None

    ihl = (ip[0] & 0x0F) * 4

    if len(ip) < ihl + 8:
        return None

    protocol = ip[9]

    # UDP
    if protocol != 17:
        return None

    udp = ip[ihl:]

    src_port, dst_port, udp_length, checksum = struct.unpack(
        "!HHHH",
        udp[:8]
    )

    if dst_port != 5500:
        return None

    if udp_length < 8:
        return None

    return udp[8:udp_length]


def decode_csi(payload):

    if len(payload) < CSI_HEADER_SIZE + NFFT * 4:
        return None

    # Nexmon CSI header
    magic = struct.unpack_from("<H", payload, 0)[0]

    if magic != 0x1111:
        return None

    # Nexmon header fields
    src_mac = payload[4:10]
    sequence = struct.unpack_from("<H", payload, 10)[0]
    core = struct.unpack_from("<H", payload, 12)[0]
    chanspec = struct.unpack_from("<H", payload, 14)[0]
    chip = struct.unpack_from("<H", payload, 16)[0]

    csi_data = payload[CSI_HEADER_SIZE:CSI_HEADER_SIZE + NFFT * 4]

    csi = []

    for i in range(NFFT):

        offset = i * 4

        real, imag = struct.unpack_from(
            "<hh",
            csi_data,
            offset
        )

        amplitude = math.sqrt(
            real * real +
            imag * imag
        )

        phase = math.atan2(imag, real)

        csi.append(
            (real, imag, amplitude, phase)
        )

    mac_string = ":".join(
        f"{b:02x}" for b in src_mac
    )

    return {
        "src_mac": mac_string,
        "sequence": sequence,
        "core": core,
        "chanspec": chanspec,
        "chip": chip,
        "csi": csi,
    }


def main():

    print("Nexmon CSI decoder")
    print("==================")
    print(f"Input : {PCAP_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    packet_count = 0
    csi_count = 0

    fieldnames = [
        "timestamp",
        "src_mac",
        "sequence",
        "core",
        "chanspec",
        "chip",
    ]

    for i in range(NFFT):
        fieldnames.append(f"sc_{i}_real")

    for i in range(NFFT):
        fieldnames.append(f"sc_{i}_imag")

    for i in range(NFFT):
        fieldnames.append(f"sc_{i}_amplitude")

    for i in range(NFFT):
        fieldnames.append(f"sc_{i}_phase")

    with open(OUTPUT_FILE, "w", newline="") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for ts_sec, ts_usec, packet in read_pcap_packets(
            PCAP_FILE
        ):

            packet_count += 1

            payload = extract_udp_payload(packet)

            if payload is None:
                continue

            decoded = decode_csi(payload)

            if decoded is None:
                continue

            csi_count += 1

            row = {
                "timestamp":
                    ts_sec + ts_usec / 1_000_000,

                "src_mac":
                    decoded["src_mac"],

                "sequence":
                    decoded["sequence"],

                "core":
                    decoded["core"],

                "chanspec":
                    decoded["chanspec"],

                "chip":
                    decoded["chip"],
            }

            for i, (real, imag, amplitude, phase) in enumerate(
                decoded["csi"]
            ):
                row[f"sc_{i}_real"] = real
                row[f"sc_{i}_imag"] = imag
                row[f"sc_{i}_amplitude"] = amplitude
                row[f"sc_{i}_phase"] = phase

            writer.writerow(row)

    print()
    print("Finished.")
    print(f"PCAP packets inspected : {packet_count}")
    print(f"CSI packets decoded    : {csi_count}")
    print(f"CSV file               : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
