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
# Calculate normalization statistics
# ============================================================

def calculate_train_statistics():

    print("\nCalculating normalization statistics")
    print("using TRAINING data only...")

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    mean = X_train.mean(
        dtype=np.float64
    )

    std = X_train.std(
        dtype=np.float64
    )

    print(
        f"Training mean: {mean:.6f}"
    )

    print(
        f"Training std:  {std:.6f}"
    )

    return np.float32(mean), np.float32(std)


# ============================================================
# CSI Dataset
# ============================================================

class CSIDataset(Dataset):

    def __init__(
        self,
        split,
        mean,
        std,
    ):

        self.X = np.load(
            DATA_DIR / f"X_{split}.npy"
        )

        self.y = np.load(
            LABEL_DIR / f"y_{split}.npy"
        )

        # ----------------------------------------------------
        # Normalize using TRAIN statistics
        # ----------------------------------------------------

        self.X = (
            self.X - mean
        ) / std

        # ----------------------------------------------------
        # Add CNN channel dimension
        #
        # (N, 250, 90)
        #      ↓
        # (N, 1, 250, 90)
        # ----------------------------------------------------

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

        print(
            f"  normalized mean: "
            f"{self.X.mean():.6f}"
        )

        print(
            f"  normalized std:  "
            f"{self.X.std():.6f}"
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
# Create DataLoaders
# ============================================================

def create_dataloaders():

    mean, std = calculate_train_statistics()

    train_dataset = CSIDataset(
        "train",
        mean,
        std,
    )

    val_dataset = CSIDataset(
        "val",
        mean,
        std,
    )

    test_dataset = CSIDataset(
        "test",
        mean,
        std,
    )

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

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        mean,
        std,
    )


# ============================================================
# Test
# ============================================================

def main():

    print("=" * 60)
    print("CSI-HAR V2 DATASET PIPELINE")
    print("=" * 60)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        mean,
        std,
    ) = create_dataloaders()

    print("\n" + "=" * 60)
    print("CLASS WEIGHTS")
    print("=" * 60)

    class_weights = calculate_class_weights(
        train_dataset
    )

    for class_id, weight in enumerate(class_weights):

        print(
            f"{class_id}: "
            f"{CLASS_NAMES[class_id]:<10} "
            f"{weight.item():.4f}"
        )

    print("\n" + "=" * 60)
    print("BATCH TEST")
    print("=" * 60)

    X_batch, y_batch = next(
        iter(train_loader)
    )

    print(
        f"X batch shape: {X_batch.shape}"
    )

    print(
        f"y batch shape: {y_batch.shape}"
    )

    print(
        f"X dtype: {X_batch.dtype}"
    )

    print(
        f"X mean: {X_batch.mean().item():.6f}"
    )

    print(
        f"X std: {X_batch.std().item():.6f}"
    )

    print(
        f"X min: {X_batch.min().item():.4f}"
    )

    print(
        f"X max: {X_batch.max().item():.4f}"
    )

    print("\n" + "=" * 60)
    print("V2 DATASET PIPELINE READY")
    print("=" * 60)


if __name__ == "__main__":
    main()