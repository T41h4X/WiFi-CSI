import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from csi_dataset import (
    CSIDataset,
    calculate_class_weights,
    BATCH_SIZE,
    CLASS_NAMES,
)


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 4

EPOCHS = 30

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "csi_har_cnn_best.pth"


# ============================================================
# CNN Model
# ============================================================

class CSI_CNN(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # Adaptive pooling
            # ------------------------------------------------

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.30),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for X, y in loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(
            outputs,
            y,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * X.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            y.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# Evaluate
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(X)

            loss = criterion(
                outputs,
                y,
            )

            running_loss += (
                loss.item() * X.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    return (
        epoch_loss,
        epoch_accuracy,
        np.array(all_labels),
        np.array(all_predictions),
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("CSI-HAR CNN BASELINE")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "Using CPU. "
            "Training will take some time."
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = CSIDataset("train")
    val_dataset = CSIDataset("val")
    test_dataset = CSIDataset("test")

    # --------------------------------------------------------
    # DataLoaders
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
    # Model
    # --------------------------------------------------------

    model = CSI_CNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Class-weighted loss
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataset
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        CLASS_NAMES,

                    "best_val_accuracy":
                        best_val_accuracy,
                },
                MODEL_PATH,
            )

            print(
                f"  -> Saved best model "
                f"(val accuracy: "
                f"{best_val_accuracy:.4f})"
            )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint['best_val_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    test_loss, test_accuracy, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    print(
        f"\nTest Loss:      {test_loss:.4f}"
    )

    print(
        f"Test Accuracy:  {test_accuracy:.4f}"
    )

    print(
        f"Macro Precision: {precision:.4f}"
    )

    print(
        f"Macro Recall:    {recall:.4f}"
    )

    print(
        f"Macro F1:        {f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")
    print()

    target_names = [
        CLASS_NAMES[i]
        for i in range(NUM_CLASSES)
    ]

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("Confusion Matrix:")

    print(cm)

    print("\nRows = Actual")
    print("Columns = Predicted")

    print(
        "\nClasses:"
    )

    for i, name in CLASS_NAMES.items():

        print(
            f"  {i} = {name}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    np.save(
        "test_confusion_matrix.npy",
        cm,
    )

    np.save(
        "test_predictions.npy",
        y_pred,
    )

    np.save(
        "test_labels.npy",
        y_true,
    )

    print("\nSaved:")
    print(f"  {MODEL_PATH}")
    print("  test_confusion_matrix.npy")
    print("  test_predictions.npy")
    print("  test_labels.npy")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()