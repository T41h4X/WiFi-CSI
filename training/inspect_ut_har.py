import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path("Data/UT_HAR")

DATA_DIR = BASE_DIR / "data"
LABEL_DIR = BASE_DIR / "label"

OUTPUT_DIR = Path("inspection_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# UT-HAR label mapping
CLASS_NAMES = {
    0: "Lying",
    1: "Fall",
    2: "Walking",
    3: "Pickup",
    4: "Running",
    5: "Sitting",
    6: "Standing",
}

# These are the four activities used in our first experiment
TARGET_CLASSES = {
    0: "Lying",
    2: "Walking",
    5: "Sitting",
    6: "Standing",
}


# ============================================================
# Load dataset
# ============================================================

print("=" * 60)
print("UT-HAR DATASET INSPECTION")
print("=" * 60)

print("\nLoading training data...")

X_train = np.load(DATA_DIR / "X_train.csv")
y_train = np.load(LABEL_DIR / "y_train.csv")

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print(f"\nData type: {X_train.dtype}")
print(f"Minimum value: {X_train.min():.4f}")
print(f"Maximum value: {X_train.max():.4f}")


# ============================================================
# Class distribution
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

unique_labels, counts = np.unique(y_train, return_counts=True)

for label, count in zip(unique_labels, counts):
    name = CLASS_NAMES.get(int(label), "Unknown")
    print(f"Label {label}: {name:<10} -> {count} samples")


# ============================================================
# Target class distribution
# ============================================================

print("\n" + "=" * 60)
print("FOUR TARGET ACTIVITIES")
print("=" * 60)

for label, name in TARGET_CLASSES.items():
    count = np.sum(y_train == label)
    print(f"Label {label}: {name:<10} -> {count} samples")


# ============================================================
# Find one sample for each target activity
# ============================================================

print("\n" + "=" * 60)
print("SELECTING EXAMPLES")
print("=" * 60)

selected_samples = {}

for label, name in TARGET_CLASSES.items():

    indices = np.where(y_train == label)[0]

    if len(indices) == 0:
        print(f"WARNING: No samples found for {name}")
        continue

    # Select a sample approximately from the middle
    index = indices[len(indices) // 2]

    selected_samples[label] = {
        "index": index,
        "data": X_train[index],
        "name": name,
    }

    print(
        f"{name:<10} -> "
        f"dataset index {index}, "
        f"shape {X_train[index].shape}"
    )


# ============================================================
# Plot individual CSI representations
# ============================================================

print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

for label, sample_info in selected_samples.items():

    sample = sample_info["data"]
    name = sample_info["name"]
    index = sample_info["index"]

    plt.figure(figsize=(12, 6))

    plt.imshow(
        sample,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    plt.colorbar(label="CSI value")

    plt.xlabel("Subcarrier / Feature")
    plt.ylabel("Time / CSI sample")
    plt.title(
        f"UT-HAR CSI Representation - {name}\n"
        f"Dataset index: {index}"
    )

    plt.tight_layout()

    output_file = OUTPUT_DIR / f"example_{name.lower()}.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_file}")


# ============================================================
# Create combined figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
)

axes = axes.flatten()

for ax, (label, sample_info) in zip(
    axes,
    selected_samples.items(),
):

    sample = sample_info["data"]
    name = sample_info["name"]
    index = sample_info["index"]

    image = ax.imshow(
        sample,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    ax.set_title(
        f"{name}\n"
        f"Index: {index}"
    )

    ax.set_xlabel("Subcarrier / Feature")
    ax.set_ylabel("Time / CSI sample")

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )


fig.suptitle(
    "UT-HAR — Four Target Activities",
    fontsize=16,
)

fig.tight_layout()

combined_file = OUTPUT_DIR / "four_target_activities.png"

fig.savefig(
    combined_file,
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)

print(f"\nSaved combined figure: {combined_file}")


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)

print(f"\nResults are located in:")
print(OUTPUT_DIR.resolve())

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  - {file.name}")

print("\nEverything completed successfully.")