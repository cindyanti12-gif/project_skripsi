import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import numpy as np

from dataset_loader import CocoDetectionDataset
from collate_fn import collate_fn

from model.detr_fusion import DETRVGGFusion
from matcher import HungarianMatcher


# =====================================
# CONFIG
# =====================================

device = torch.device("cpu")

CHECKPOINT_DIR = "checkpoints/run_20260618_175319"

MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

CONF_THRESHOLD = 0.5

SAVE_DIR = "evaluation"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# =====================================
# DATASET
# =====================================

print("=" * 50)
print("Loading Test Dataset...")
print("=" * 50)

test_dataset = CocoDetectionDataset(
    image_dir="dataset/test",
    annotation_file="dataset/test/_annotations.coco.json"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)

print(
    f"Test Dataset : {len(test_dataset)}"
)


# =====================================
# MODEL
# =====================================

print("\nLoading Model...")

model = DETRVGGFusion(
    num_classes=2,
    num_queries=25
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model Loaded Successfully")


# =====================================
# MATCHER
# =====================================

matcher = HungarianMatcher()


# =====================================
# EVALUATION
# =====================================

all_targets = []
all_predictions = []

print("\nRunning Evaluation...\n")

with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(device)

        outputs = model(images)

        pred_logits = outputs["pred_logits"]

        pred_boxes = outputs["pred_boxes"]

        probs = F.softmax(
            pred_logits,
            dim=-1
        )

        scores, pred_labels = probs.max(
            dim=-1
        )

        indices = matcher(
            outputs,
            targets
        )

        for b, (pred_idx, tgt_idx) in enumerate(indices):

            target_labels = targets[b][
                "class_labels"
            ]

            matched_pred_labels = pred_labels[
                b
            ][pred_idx]

            matched_scores = scores[
                b
            ][pred_idx]

            for gt, pred, score in zip(
                target_labels[tgt_idx],
                matched_pred_labels,
                matched_scores
            ):

                gt = int(gt.item())

                if score.item() < CONF_THRESHOLD:

                    pred = 0

                else:

                    pred = int(pred.item())

                all_targets.append(gt)
                all_predictions.append(pred)


# =====================================
# METRICS
# =====================================

precision = precision_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

print("=" * 50)
print("RESULT")
print("=" * 50)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# =====================================
# SAVE METRICS
# =====================================

metrics_file = os.path.join(
    SAVE_DIR,
    "metrics.txt"
)

with open(
    metrics_file,
    "w"
) as f:

    f.write(
        f"Precision : {precision:.4f}\n"
    )

    f.write(
        f"Recall    : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score  : {f1:.4f}\n"
    )


# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=[1, 2]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "parasite",
        "non_parasite"
    ]
)

fig, ax = plt.subplots(
    figsize=(6, 6)
)

disp.plot(
    ax=ax
)

plt.title(
    "Confusion Matrix"
)

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "confusion_matrix.png"
    )
)

plt.close()


# =====================================
# DONE
# =====================================

print("\nSaved:")

print(
    os.path.join(
        SAVE_DIR,
        "metrics.txt"
    )
)

print(
    os.path.join(
        SAVE_DIR,
        "confusion_matrix.png"
    )
)

print("\nEvaluation Finished")