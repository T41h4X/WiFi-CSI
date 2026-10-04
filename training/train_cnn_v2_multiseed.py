import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from csi_dataset_v2 import create_dataloaders


# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [42, 123, 456, 789, 2026]

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

EPOCHS = 50

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

PATIENCE = 8

RESULTS_FILE = "csi_har_cnn_v2_multiseed_results.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_labels = []


    for X, y in loader:

        X = X.to(device)

        y = y.to(device)


        optimizer.zero_grad()


        outputs = model(X)


        loss = criterion(
            outputs,
            y
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item() * X.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
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


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )


    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)

            y = y.to(device)


            outputs = model(X)


            loss = criterion(
                outputs,
                y
            )


            running_loss += (
                loss.item() * X.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            all_predictions.extend(
                predictions.cpu()
                .numpy()
            )


            all_labels.extend(
                y.cpu()
                .numpy()
            )


    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )


    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# SINGLE SEED EXPERIMENT
# ============================================================

def run_experiment(seed):

    print()
    print("=" * 70)
    print(f"STARTING EXPERIMENT - SEED {seed}")
    print("=" * 70)

    set_seed(seed)

    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print()


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

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

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    print()


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = CSI_CNN_V2(
        num_classes=4
    ).to(device)


    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=device
    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )


    # --------------------------------------------------------
    # BEST MODEL TRACKING
    # --------------------------------------------------------

    best_val_f1 = -1.0

    best_state_dict = None

    best_epoch = 0

    epochs_without_improvement = 0


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        (
            train_loss,
            train_acc,
            train_precision,
            train_recall,
            train_f1
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )


        (
            val_loss,
            val_acc,
            val_precision,
            val_recall,
            val_f1
        ) = evaluate(
            model,
            val_loader,
            criterion
        )


        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        scheduler.step(
            val_f1
        )


        print(
            f"Seed {seed} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"LR {current_lr:.6f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Train F1 {train_f1:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_epoch = epoch

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0


        else:

            epochs_without_improvement += 1


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"Early stopping after "
                f"{PATIENCE} epochs without "
                f"improvement."
            )

            break


    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    model.load_state_dict(
        best_state_dict
    )


    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1
    ) = evaluate(
        model,
        test_loader,
        criterion
    )


    print()
    print("-" * 70)
    print(f"SEED {seed} RESULTS")
    print("-" * 70)

    print(
        f"Best Epoch:           {best_epoch}"
    )

    print(
        f"Best Validation F1:   {best_val_f1:.4f}"
    )

    print(
        f"Test Accuracy:        {test_accuracy:.4f}"
    )

    print(
        f"Test Macro Precision: {test_precision:.4f}"
    )

    print(
        f"Test Macro Recall:    {test_recall:.4f}"
    )

    print(
        f"Test Macro F1:        {test_f1:.4f}"
    )


    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CSI-HAR CNN V2 MULTI-SEED EXPERIMENT")
    print("=" * 70)

    print(f"Device: {device}")

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Number of runs: {len(SEEDS)}"
    )

    print()


    results = []


    # --------------------------------------------------------
    # RUN ALL SEEDS
    # --------------------------------------------------------

    for seed in SEEDS:

        result = run_experiment(
            seed
        )

        results.append(
            result
        )


    # --------------------------------------------------------
    # CALCULATE SUMMARY STATISTICS
    # --------------------------------------------------------

    accuracies = np.array([
        result["test_accuracy"]
        for result in results
    ])

    precisions = np.array([
        result["test_precision"]
        for result in results
    ])

    recalls = np.array([
        result["test_recall"]
        for result in results
    ])

    f1_scores = np.array([
        result["test_f1"]
        for result in results
    ])


    accuracy_mean = accuracies.mean()

    accuracy_std = accuracies.std(
        ddof=1
    )


    precision_mean = precisions.mean()

    precision_std = precisions.std(
        ddof=1
    )


    recall_mean = recalls.mean()

    recall_std = recalls.std(
        ddof=1
    )


    f1_mean = f1_scores.mean()

    f1_std = f1_scores.std(
        ddof=1
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MULTI-SEED RESULTS")
    print("=" * 70)


    print()

    print(
        f"{'Seed':<10}"
        f"{'Accuracy':<15}"
        f"{'Macro F1':<15}"
    )

    print("-" * 40)


    for result in results:

        print(
            f"{result['seed']:<10}"
            f"{result['test_accuracy']:.4f}"
            f"{'':<9}"
            f"{result['test_f1']:.4f}"
        )


    print("-" * 40)


    print(
        f"{'Mean':<10}"
        f"{accuracy_mean:.4f}"
        f"{'':<9}"
        f"{f1_mean:.4f}"
    )


    print(
        f"{'Std':<10}"
        f"{accuracy_std:.4f}"
        f"{'':<9}"
        f"{f1_std:.4f}"
    )


    print()
    print("=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)


    print(
        f"Accuracy:        "
        f"{accuracy_mean:.4f} ± "
        f"{accuracy_std:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{precision_mean:.4f} ± "
        f"{precision_std:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{recall_mean:.4f} ± "
        f"{recall_std:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{f1_mean:.4f} ± "
        f"{f1_std:.4f}"
    )


    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    with open(
        RESULTS_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)


        writer.writerow([
            "seed",
            "best_epoch",
            "best_val_f1",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1"
        ])


        for result in results:

            writer.writerow([
                result["seed"],
                result["best_epoch"],
                f"{result['best_val_f1']:.6f}",
                f"{result['test_accuracy']:.6f}",
                f"{result['test_precision']:.6f}",
                f"{result['test_recall']:.6f}",
                f"{result['test_f1']:.6f}"
            ])


    print()
    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )

    print()
    print("=" * 70)
    print("MULTI-SEED EXPERIMENT COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()