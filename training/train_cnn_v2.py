import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from csi_dataset_v2 import create_dataloaders


CLASS_NAMES = [
    "Lying",
    "Walking",
    "Sitting",
    "Standing"
]

CLASS_WEIGHTS = [
    1.0762,
    0.4821,
    1.7656,
    2.3251
]

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 50

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

PATIENCE = 8

MODEL_PATH = "csi_har_cnn_v2_best.pth"
CONFUSION_MATRIX_PATH = "csi_har_cnn_v2_confusion_matrix.png"
CURVES_PATH = "csi_har_cnn_v2_training_curves.png"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CSI-HAR CNN V2 TRAINING")
print("=" * 60)

print(f"Device: {device}")
print(f"Random seed: {SEED}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print()


# ============================================================
# DATA
# ============================================================

(
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader,
    normalization_mean,
    normalization_std,
) = create_dataloaders()


print("Datasets loaded.")
print(f"Training batches:   {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches:       {len(test_loader)}")
print()


# ============================================================
# MODEL
# ============================================================

class CSI_CNN_V2(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = CSI_CNN_V2(num_classes=4).to(device)


print("MODEL")
print("-" * 60)
print(model)
print()


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights = torch.tensor(
    CLASS_WEIGHTS,
    dtype=torch.float32,
    device=device
)

print("Class weights:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name:<10} {CLASS_WEIGHTS[i]:.4f}")

print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * X.size(0)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            y.detach().cpu().numpy()
        )

    epoch_loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return epoch_loss, accuracy, precision, recall, f1


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            outputs = model(X)

            loss = criterion(outputs, y)

            running_loss += loss.item() * X.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    epoch_loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return epoch_loss, accuracy, precision, recall, f1


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {

    "train_loss": [],
    "train_accuracy": [],
    "train_f1": [],

    "val_loss": [],
    "val_accuracy": [],
    "val_f1": [],

    "learning_rate": []
}


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_f1 = -1.0

epochs_without_improvement = 0

print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc, train_precision, train_recall, train_f1 = \
        train_one_epoch(model, train_loader)

    val_loss, val_acc, val_precision, val_recall, val_f1 = \
        evaluate(model, val_loader)

    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)
    history["train_f1"].append(train_f1)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)
    history["val_f1"].append(val_f1)

    history["learning_rate"].append(current_lr)

    scheduler.step(val_f1)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"LR {current_lr:.6f} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Train F1 {train_f1:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_acc:.4f} | "
        f"Val F1 {val_f1:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "val_accuracy": val_acc,
                "seed": SEED,
            },
            MODEL_PATH
        )

        epochs_without_improvement = 0

        print(
            f"  -> New best model saved! "
            f"Validation Macro-F1: {val_f1:.4f}"
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            f"Early stopping triggered after "
            f"{PATIENCE} epochs without improvement."
        )

        break


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Best validation Macro-F1: {best_val_f1:.4f}")
print(f"Best model saved to: {MODEL_PATH}")
print()


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Loaded best model from epoch "
    f"{checkpoint['epoch']}"
)

print()


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

model.eval()

all_test_predictions = []
all_test_labels = []

test_loss = 0.0

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(device)
        y = y.to(device)

        outputs = model(X)

        loss = criterion(outputs, y)

        test_loss += loss.item() * X.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_test_predictions.extend(
            predictions.cpu().numpy()
        )

        all_test_labels.extend(
            y.cpu().numpy()
        )


test_loss /= len(test_loader.dataset)

test_accuracy = accuracy_score(
    all_test_labels,
    all_test_predictions
)

test_precision, test_recall, test_f1, _ = \
    precision_recall_fscore_support(
        all_test_labels,
        all_test_predictions,
        average="macro",
        zero_division=0
    )


# ============================================================
# TEST RESULTS
# ============================================================

print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Loss:              {test_loss:.4f}")
print(f"Test Accuracy:          {test_accuracy:.4f}")
print(f"Test Macro Precision:   {test_precision:.4f}")
print(f"Test Macro Recall:      {test_recall:.4f}")
print(f"Test Macro F1:          {test_f1:.4f}")

print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_test_labels,
        all_test_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_test_labels,
    all_test_predictions
)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)
print()


# ============================================================
# SAVE CONFUSION MATRIX FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

disp.plot(
    ax=ax,
    values_format="d",
    cmap="Blues"
)

ax.set_title(
    "CSI-HAR CNN V2 - Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200
)

plt.close()


# ============================================================
# TRAINING CURVES
# ============================================================

epochs_completed = range(
    1,
    len(history["train_loss"]) + 1
)


# ------------------------------------------------------------
# LOSS CURVE
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    epochs_completed,
    history["train_loss"],
    label="Training Loss"
)

ax.plot(
    epochs_completed,
    history["val_loss"],
    label="Validation Loss"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("CNN V2 Training and Validation Loss")

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "csi_har_cnn_v2_loss_curve.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# ACCURACY CURVE
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    epochs_completed,
    history["train_accuracy"],
    label="Training Accuracy"
)

ax.plot(
    epochs_completed,
    history["val_accuracy"],
    label="Validation Accuracy"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("CNN V2 Training and Validation Accuracy")

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "csi_har_cnn_v2_accuracy_curve.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# MACRO F1 CURVE
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    epochs_completed,
    history["train_f1"],
    label="Training Macro F1"
)

ax.plot(
    epochs_completed,
    history["val_f1"],
    label="Validation Macro F1"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Macro F1")
ax.set_title("CNN V2 Training and Validation Macro F1")

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "csi_har_cnn_v2_f1_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

np.save(
    "csi_har_cnn_v2_test_predictions.npy",
    np.array(all_test_predictions)
)

np.save(
    "csi_har_cnn_v2_test_labels.npy",
    np.array(all_test_labels)
)

np.save(
    "csi_har_cnn_v2_confusion_matrix.npy",
    cm
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 60)
print("FILES GENERATED")
print("=" * 60)

print(f"Best model:")
print(f"  {MODEL_PATH}")

print()
print("Plots:")
print("  csi_har_cnn_v2_loss_curve.png")
print("  csi_har_cnn_v2_accuracy_curve.png")
print("  csi_har_cnn_v2_f1_curve.png")
print("  csi_har_cnn_v2_confusion_matrix.png")

print()
print("Prediction files:")
print("  csi_har_cnn_v2_test_predictions.npy")
print("  csi_har_cnn_v2_test_labels.npy")
print("  csi_har_cnn_v2_confusion_matrix.npy")

print()
print("=" * 60)
print("CNN V2 COMPLETE")
print("=" * 60)