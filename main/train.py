import os
import sys
import csv
import time
import datetime

import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

# ======================================================
# PROJECT PATH
# ======================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT_DIR)

# ======================================================
# IMPORT PROJECT
# ======================================================

from dataset_loader import CocoDetectionDataset
from collate_fn import collate_fn

from model.detr_fusion import DETRVGGFusion
from loss.detr_loss import DETRLoss

# ======================================================
# TRAINING CONFIGURATION
# ======================================================

NUM_EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# CREATE SAVE DIRECTORY
# ======================================================

timestamp = datetime.datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

save_dir = os.path.join(
    ROOT_DIR,
    "checkpoints",
    f"run_{timestamp}"
)

os.makedirs(
    save_dir,
    exist_ok=True
)

# ======================================================
# SAVE CONFIGURATION
# ======================================================

config_path = os.path.join(
    save_dir,
    "config.txt"
)

with open(config_path, "w") as f:

    f.write("========== TRAINING CONFIG ==========\n\n")

    f.write(f"Date           : {timestamp}\n")
    f.write(f"Device         : {DEVICE}\n")
    f.write(f"Epoch          : {NUM_EPOCHS}\n")
    f.write(f"Batch Size     : {BATCH_SIZE}\n")
    f.write(f"Learning Rate  : {LEARNING_RATE}\n")
    f.write(f"Weight Decay   : {WEIGHT_DECAY}\n")

# ======================================================
# DATASET
# ======================================================

print("=" * 50)
print("Loading Dataset...")
print("=" * 50)

train_dataset = CocoDetectionDataset(
    image_dir="dataset/train",
    annotation_file="dataset/train/_annotations.coco.json"
)

val_dataset = CocoDetectionDataset(
    image_dir="dataset/val",
    annotation_file="dataset/val/_annotations.coco.json"
)

print(f"Train Dataset      : {len(train_dataset)}")
print(f"Validation Dataset : {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# ======================================================
# MODEL
# ======================================================

print("\nBuilding Model...")

model = DETRVGGFusion(
    num_classes=2,
    num_queries=25
)

model.to(DEVICE)

criterion = DETRLoss()
criterion.to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("Model Loaded Successfully.")

# ======================================================
# HISTORY
# ======================================================

history = []

best_val_loss = float("inf")

start_time = time.time()

log_path = os.path.join(
    save_dir,
    "training_log.txt"
)

log_file = open(
    log_path,
    "w"
)

print("\nStart Training...\n")

# ======================================================
# TRAINING LOOP
# ======================================================

for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()

    # ===========================
    # TRAIN
    # ===========================

    model.train()

    train_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
        leave=True
    )

    for images, targets in progress:

        images = images.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        outputs = model(images)

        loss_dict = criterion(
            outputs,
            targets
        )

        loss = loss_dict["loss"]

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss /= len(train_loader)

    # ===========================
    # VALIDATION
    # ===========================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        val_progress = tqdm(
            val_loader,
            desc="Validation",
            leave=False
        )

        for images, targets in val_progress:

            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            outputs = model(images)

            loss_dict = criterion(
                outputs,
                targets
            )

            loss = loss_dict["loss"]

            val_loss += loss.item()

    val_loss /= len(val_loader)

    # ===========================
    # SAVE HISTORY
    # ===========================

    history.append([
        epoch + 1,
        train_loss,
        val_loss
    ])

    # ===========================
    # SAVE LAST MODEL
    # ===========================

    torch.save(
        model.state_dict(),
        os.path.join(
            save_dir,
            "last_model.pth"
        )
    )

    # ===========================
    # SAVE BEST MODEL
    # ===========================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            os.path.join(
                save_dir,
                "best_model.pth"
            )
        )

    # ===========================
    # TIME
    # ===========================

    epoch_time = time.time() - epoch_start

    total_time = time.time() - start_time

    remain = epoch_time * (NUM_EPOCHS - epoch - 1)

    remain_hour = int(remain // 3600)

    remain_min = int((remain % 3600) // 60)

    remain_sec = int(remain % 60)

    # ===========================
    # PRINT
    # ===========================

    message = (
        f"\n"
        f"Epoch {epoch+1}/{NUM_EPOCHS}\n"
        f"--------------------------------------\n"
        f"Train Loss : {train_loss:.4f}\n"
        f"Val Loss   : {val_loss:.4f}\n"
        f"Best Loss  : {best_val_loss:.4f}\n"
        f"Epoch Time : {epoch_time/60:.2f} min\n"
        f"Elapsed    : {total_time/60:.2f} min\n"
        f"ETA        : {remain_hour:02d}:{remain_min:02d}:{remain_sec:02d}\n"
    )

    print(message)

    log_file.write(message)

    log_file.flush()

# ======================================================
# TRAINING FINISHED
# ======================================================

log_file.close()

total_training_time = time.time() - start_time

# ======================================================
# SAVE HISTORY CSV
# ======================================================

csv_path = os.path.join(
    save_dir,
    "train_history.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Epoch",
        "Train Loss",
        "Validation Loss"
    ])

    writer.writerows(history)

# ======================================================
# SAVE LOSS CURVE
# ======================================================

epochs = [row[0] for row in history]

train_losses = [row[1] for row in history]

val_losses = [row[2] for row in history]

plt.figure(figsize=(10,6))

plt.plot(
    epochs,
    train_losses,
    marker='o',
    linewidth=2,
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker='o',
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch", fontsize=12)

plt.ylabel("Loss", fontsize=12)

plt.title("Training History")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        save_dir,
        "loss_curve.png"
    ),
    dpi=300
)

plt.close()

# ======================================================
# UPDATE CONFIG
# ======================================================

with open(
    config_path,
    "a"
) as f:

    f.write("\n")
    f.write("========== RESULT ==========\n\n")

    f.write(f"Best Validation Loss : {best_val_loss:.4f}\n")

    f.write(f"Training Time        : {total_training_time/60:.2f} minutes\n")

    f.write(f"Train Dataset        : {len(train_dataset)}\n")

    f.write(f"Validation Dataset   : {len(val_dataset)}\n")

# ======================================================
# FINISH
# ======================================================

print("\n")

print("=" * 60)

print("           TRAINING FINISHED SUCCESSFULLY")

print("=" * 60)

print(f"Best Validation Loss : {best_val_loss:.4f}")

print(f"Training Time        : {total_training_time/60:.2f} minutes")

print()

print("Files Generated")

print("------------------------------")

print("✓ best_model.pth")

print("✓ last_model.pth")

print("✓ train_history.csv")

print("✓ training_log.txt")

print("✓ loss_curve.png")

print("✓ config.txt")

print()

print(f"Saved in : {save_dir}")

print("=" * 60)