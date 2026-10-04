import csv
import math

FILE = "/home/fydp/kftn_csi.csv"

SUBCARRIERS = [1, 2, 5, 10, 15, 20, 25, 27, 36, 40, 45, 50, 55, 60, 63]

values = {sc: [] for sc in SUBCARRIERS}
timestamps = []

with open(FILE, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        timestamps.append(float(row["timestamp"]))

        for sc in SUBCARRIERS:
            values[sc].append(float(row[f"sc_{sc}_amplitude"]))


print()
print("=" * 65)
print("              TEMPORAL CSI CHECK")
print("=" * 65)

print(f"\nFrames: {len(timestamps)}")

if len(timestamps) > 1:
    duration = timestamps[-1] - timestamps[0]
    print(f"Duration: {duration:.2f} seconds")
    print(f"Frame rate: {len(timestamps) / duration:.2f} CSI/s")

print()
print("Subcarrier | Mean | Std Dev | Min | Max")
print("-" * 55)

for sc in SUBCARRIERS:
    x = values[sc]

    mean = sum(x) / len(x)

    variance = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(variance)

    print(
        f"{sc:10d} | "
        f"{mean:7.2f} | "
        f"{std:8.2f} | "
        f"{min(x):7.2f} | "
        f"{max(x):7.2f}"
    )

print()
print("=" * 65)
print("Interpretation:")
print("Higher Std Dev = more temporal variation.")
print("Very low Std Dev = relatively stable subcarrier.")
print("=" * 65)
print()
