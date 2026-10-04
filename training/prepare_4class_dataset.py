import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path("Data/UT_HAR")

DATA_DIR = BASE_DIR / "data"
LABEL_DIR = BASE_DIR / "label"

OUTPUT_DIR = Path("Data/UT_HAR_4CLASS")

OUTPUT_DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_LABEL_DIR = OUTPUT_DIR / "label"

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Original UT-HAR labels
# ============================================================

# Original UT-HAR:
#
# 0 -> lie down
# 1 -> fall
# 2 -> walk
# 3 -> pickup
# 4 -> run
# 5 -> sit down
# 6 -> stand up


# New labels for our first experiment:
#
# 0 -> Lying
# 1 -> Walking
# 2 -> Sitting
# 3 -> Standing

LABEL_MAPPING = {
    0: 0,  # Lying
    2: 1,  # Walking
    5: 2,  # Sitting
    6: 3,  # Standing
}

CLASS_NAMES = {
    0: "Lying",
    1: "Walking",
    2: "Sitting",
    3: "Standing",
}


# ============================================================
# Function to prepare one split
# ============================================================

def prepare_split(split_name):

    print("\n" + "=" * 60)
    print(f"PROCESSING {split_name.upper()} SPLIT")
    print("=" * 60)

    data_file = DATA_DIR / f"X_{split_name}.csv"
    label_file = LABEL_DIR / f"y_{split_name}.csv"

    print(f"Loading data:   {data_file}")
    print(f"Loading labels: {label_file}")

    X = np.load(data_file)
    y = np.load(label_file)

    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")

    # --------------------------------------------------------
    # Select only our four target classes
    # --------------------------------------------------------

    target_mask = np.isin(
        y,
        list(LABEL_MAPPING.keys())
    )

    X_selected = X[target_mask]
    y_selected_original = y[target_mask]

    # --------------------------------------------------------
    # Convert original labels to new 0-3 labels
    # --------------------------------------------------------

    y_selected = np.array(
        [
            LABEL_MAPPING[int(label)]
            for label in y_selected_original
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Print distribution
    # --------------------------------------------------------

    print("\nSelected samples:")

    for new_label, class_name in CLASS_NAMES.items():

        count = np.sum(y_selected == new_label)

        print(
            f"  {new_label} -> "
            f"{class_name:<10}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # Save the processed split
    # --------------------------------------------------------

    output_data_file = (
        OUTPUT_DATA_DIR /
        f"X_{split_name}.npy"
    )

    output_label_file = (
        OUTPUT_LABEL_DIR /
        f"y_{split_name}.npy"
    )

    np.save(
        output_data_file,
        X_selected.astype(np.float32),
    )

    np.save(
        output_label_file,
        y_selected,
    )

    print("\nSaved:")
    print(f"  {output_data_file}")
    print(f"  {output_label_file}")

    print(
        f"New X shape: {X_selected.shape}"
    )

    print(
        f"New y shape: {y_selected.shape}"
    )

    return X_selected, y_selected


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("UT-HAR -> FOUR CLASS DATASET PREPARATION")
    print("=" * 60)

    print("\nTarget activities:")

    for label, name in CLASS_NAMES.items():
        print(f"  {label} -> {name}")

    print(
        "\nOriginal dataset files will NOT be modified."
    )

    # Process all three official splits
    train_X, train_y = prepare_split("train")
    val_X, val_y = prepare_split("val")
    test_X, test_y = prepare_split("test")

    # ========================================================
    # Final summary
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    print(
        f"\nTrain: {train_X.shape[0]} samples"
    )

    print(
        f"Validation: {val_X.shape[0]} samples"
    )

    print(
        f"Test: {test_X.shape[0]} samples"
    )

    print(
        f"\nInput shape: "
        f"{train_X.shape[1:]}"
    )

    print("\nClasses:")

    for label, name in CLASS_NAMES.items():

        train_count = np.sum(train_y == label)
        val_count = np.sum(val_y == label)
        test_count = np.sum(test_y == label)

        print(
            f"  {label}: {name:<10} "
            f"Train={train_count:<4} "
            f"Val={val_count:<3} "
            f"Test={test_count:<3}"
        )

    print("\nOutput directory:")
    print(OUTPUT_DIR.resolve())

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()