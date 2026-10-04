import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("Data/UT_HAR_4CLASS")

DATA_DIR = DATASET_DIR / "data"
LABEL_DIR = DATASET_DIR / "label"

BATCH_SIZE = 32

CLASS_NAMES = {
    0: "Lying",
    1: "Walking",
    2: "Sitting",
    3: "Standing",
}


# ============================================================
# CSI Dataset
# ============================================================

class CSIDataset(Dataset):

    def __init__(self, split):

        self.X = np.load(
            DATA_DIR / f"X_{split}.npy"
        )

        self.y = np.load(
            LABEL_DIR / f"y_{split}.npy"
        )

        # Convert from:
        # (N, 250, 90)
        #
        # to:
        # (N, 1, 250, 90)
        self.X = self.X[:, np.newaxis, :, :]

        print(
            f"{split.upper()} dataset loaded:"
        )

        print(
            f"  X shape: {self.X.shape}"
        )

        print(
            f"  y shape: {self.y.shape}"
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        x = torch.from_numpy(
            self.X[index]
        ).float()

        y = torch.tensor(
            self.y[index]
        ).long()

        return x, y


# ============================================================
# Calculate class weights
# ============================================================

def calculate_class_weights(dataset):

    labels = dataset.y

    class_counts = np.bincount(
        labels,
        minlength=len(CLASS_NAMES),
    )

    total = len(labels)

    num_classes = len(CLASS_NAMES)

    weights = total / (
        num_classes * class_counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# Main test
# ============================================================

def main():

    print("=" * 60)
    print("CSI-HAR PYTORCH DATASET TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_dataset = CSIDataset("train")
    val_dataset = CSIDataset("val")
    test_dataset = CSIDataset("test")

    # --------------------------------------------------------
    # Calculate class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataset
    )

    print("\nClass weights:")

    for class_id, weight in enumerate(class_weights):

        print(
            f"  {class_id}: "
            f"{CLASS_NAMES[class_id]:<10} "
            f"{weight.item():.4f}"
        )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Inspect one batch
    # --------------------------------------------------------

    print("\nLoading one training batch...")

    X_batch, y_batch = next(
        iter(train_loader)
    )

    print(
        f"Batch X shape: {X_batch.shape}"
    )

    print(
        f"Batch y shape: {y_batch.shape}"
    )

    print(
        f"Batch X dtype: {X_batch.dtype}"
    )

    print(
        f"Batch y dtype: {y_batch.dtype}"
    )

    print(
        f"Batch X min: {X_batch.min().item():.4f}"
    )

    print(
        f"Batch X max: {X_batch.max().item():.4f}"
    )

    # --------------------------------------------------------
    # Verify labels
    # --------------------------------------------------------

    print("\nExample batch labels:")

    print(
        y_batch.tolist()
    )

    print("\nExample batch activities:")

    print(
        [
            CLASS_NAMES[int(label)]
            for label in y_batch[:10]
        ]
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET PIPELINE TEST COMPLETE")
    print("=" * 60)

    print(
        f"\nTrain batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    print(
        f"Test batches: {len(test_loader)}"
    )

    print(
        "\nPyTorch is ready for CNN training."
    )


if __name__ == "__main__":
    main()