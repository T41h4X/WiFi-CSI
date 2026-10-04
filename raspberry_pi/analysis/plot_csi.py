import csv
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FILE = "/home/fydp/kftn_csi.csv"

# ------------------------------------------------------------
# Load CSI amplitude data
# ------------------------------------------------------------

timestamps = []
amplitude_rows = []

with open(FILE, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        timestamps.append(float(row["timestamp"]))

        amplitudes = [
            float(row[f"sc_{i}_amplitude"])
            for i in range(64)
        ]

        amplitude_rows.append(amplitudes)


timestamps = np.array(timestamps)
amplitudes = np.array(amplitude_rows)

print()
print("Loaded CSI data")
print("----------------")
print(f"Frames       : {len(timestamps)}")
print(f"Subcarriers  : {amplitudes.shape[1]}")
print(f"Duration     : {timestamps[-1] - timestamps[0]:.2f} seconds")


# Convert timestamps to seconds from beginning
time = timestamps - timestamps[0]


# ------------------------------------------------------------
# Remove known invalid/null/guard regions for ML visualization
#
# We keep the original data untouched.
# This mask is only used for the plots.
# ------------------------------------------------------------

valid_subcarriers = [
    i for i in range(64)
    if i not in [0, 28, 29, 30, 31, 32, 33, 34, 35]
]

clean_amplitudes = amplitudes[:, valid_subcarriers]


# ------------------------------------------------------------
# Downsample for plotting
# ------------------------------------------------------------

MAX_POINTS = 5000

if len(time) > MAX_POINTS:
    step = math.ceil(len(time) / MAX_POINTS)

    plot_time = time[::step]
    plot_amplitudes = amplitudes[::step]
else:
    plot_time = time
    plot_amplitudes = amplitudes


# ============================================================
# PLOT 1 — Selected CSI subcarriers over time
# ============================================================

selected = [1, 5, 10, 15, 20, 25, 36, 40, 45, 50, 55, 60]

plt.figure(figsize=(14, 7))

for sc in selected:
    plt.plot(
        plot_time,
        plot_amplitudes[:, sc],
        linewidth=0.8,
        label=f"SC {sc}"
    )

plt.xlabel("Time (seconds)")
plt.ylabel("CSI amplitude")
plt.title("CSI Amplitude Over Time")
plt.legend(ncol=4)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "/home/fydp/csi_amplitude.png",
    dpi=150
)

plt.close()

print("Saved: /home/fydp/csi_amplitude.png")


# ============================================================
# PLOT 2 — CSI Heatmap
# ============================================================

# Use only valid subcarriers
heatmap_data = clean_amplitudes[::max(1, math.ceil(len(clean_amplitudes) / 2000))]

plt.figure(figsize=(15, 8))

plt.imshow(
    heatmap_data.T,
    aspect="auto",
    origin="lower",
    interpolation="nearest"
)

plt.xlabel("Time samples")
plt.ylabel("Valid subcarrier index")
plt.title("CSI Amplitude Heatmap")

plt.colorbar(label="CSI amplitude")

plt.tight_layout()

plt.savefig(
    "/home/fydp/csi_heatmap.png",
    dpi=150
)

plt.close()

print("Saved: /home/fydp/csi_heatmap.png")


# ============================================================
# PLOT 3 — Amplitude distributions
# ============================================================

distribution_subcarriers = [1, 5, 10, 15, 20, 25, 36, 40, 45, 50, 55, 60]

plt.figure(figsize=(14, 7))

for sc in distribution_subcarriers:
    plt.hist(
        amplitudes[:, sc],
        bins=100,
        density=True,
        histtype="step",
        linewidth=1.2,
        label=f"SC {sc}"
    )

plt.xlabel("CSI amplitude")
plt.ylabel("Density")
plt.title("CSI Amplitude Distributions")
plt.legend(ncol=4)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "/home/fydp/csi_distribution.png",
    dpi=150
)

plt.close()

print("Saved: /home/fydp/csi_distribution.png")


print()
print("All plots generated successfully.")
print()
