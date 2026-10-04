import csv
from collections import Counter

FILE = "/home/fydp/kftn_csi.csv"

timestamps = []
sequences = []
cores = []
chanspecs = []
chips = []
macs = []

amp_min = [float("inf")] * 64
amp_max = [float("-inf")] * 64
amp_sum = [0.0] * 64

packet_count = 0

with open(FILE, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        packet_count += 1

        timestamps.append(float(row["timestamp"]))
        sequences.append(int(row["sequence"]))
        cores.append(int(row["core"]))
        chanspecs.append(int(row["chanspec"]))
        chips.append(int(row["chip"]))
        macs.append(row["src_mac"])

        for i in range(64):
            value = float(row[f"sc_{i}_amplitude"])

            if value < amp_min[i]:
                amp_min[i] = value

            if value > amp_max[i]:
                amp_max[i] = value

            amp_sum[i] += value


print()
print("=" * 60)
print("              CSI DATA SUMMARY")
print("=" * 60)

print(f"\nCSI frames decoded : {packet_count}")

if packet_count > 0:
    first_time = timestamps[0]
    last_time = timestamps[-1]
    duration = last_time - first_time

    print(f"First timestamp    : {first_time}")
    print(f"Last timestamp     : {last_time}")
    print(f"Capture duration   : {duration:.3f} seconds")

    if duration > 0:
        print(f"Average frame rate : {packet_count / duration:.3f} CSI/s")
        print(f"Average interval   : {duration / (packet_count - 1):.6f} seconds")

print()
print("=" * 60)
print("                    METADATA")
print("=" * 60)

print(f"\nUnique MACs        : {len(set(macs))}")
print(f"MAC addresses      : {sorted(set(macs))}")

print(f"\nUnique sequences   : {len(set(sequences))}")
print(f"Unique cores       : {sorted(set(cores))}")
print(f"Unique chanspecs   : {sorted(set(chanspecs))}")
print(f"Unique chips       : {sorted(set(chips))}")

print("\nMost common sequence values:")

for value, count in Counter(sequences).most_common(10):
    print(f"  {value:5d} : {count:6d} frames")


print()
print("=" * 60)
print("          SUBCARRIER AMPLITUDE STATISTICS")
print("=" * 60)

print()
print("Subcarrier |       Min |       Max |      Mean")
print("-" * 48)

for i in range(64):
    mean = amp_sum[i] / packet_count

    print(
        f"{i:10d} | "
        f"{amp_min[i]:9.2f} | "
        f"{amp_max[i]:9.2f} | "
        f"{mean:9.2f}"
    )


print()
print("=" * 60)
print("                 END OF ANALYSIS")
print("=" * 60)
print()
