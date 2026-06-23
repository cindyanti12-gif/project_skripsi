import torch
import torch.nn as nn
import torch.nn.functional as F

def box_cxcywh_to_xyxy(boxes):

    cx, cy, w, h = boxes.unbind(-1)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return torch.stack(
        [x1, y1, x2, y2],
        dim=-1
    )

from matcher import HungarianMatcher

def generalized_box_iou(
    boxes1,
    boxes2
):

    x1 = torch.max(
        boxes1[:, 0],
        boxes2[:, 0]
    )

    y1 = torch.max(
        boxes1[:, 1],
        boxes2[:, 1]
    )

    x2 = torch.min(
        boxes1[:, 2],
        boxes2[:, 2]
    )

    y2 = torch.min(
        boxes1[:, 3],
        boxes2[:, 3]
    )

    inter = (
        (x2 - x1).clamp(min=0)
        *
        (y2 - y1).clamp(min=0)
    )

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0])
        *
        (boxes1[:, 3] - boxes1[:, 1])
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0])
        *
        (boxes2[:, 3] - boxes2[:, 1])
    )

    union = area1 + area2 - inter

    iou = inter / (union + 1e-6)

    cx1 = torch.min(
        boxes1[:, 0],
        boxes2[:, 0]
    )

    cy1 = torch.min(
        boxes1[:, 1],
        boxes2[:, 1]
    )

    cx2 = torch.max(
        boxes1[:, 2],
        boxes2[:, 2]
    )

    cy2 = torch.max(
        boxes1[:, 3],
        boxes2[:, 3]
    )

    c_area = (
        (cx2 - cx1)
        *
        (cy2 - cy1)
    )

    giou = iou - (
        (c_area - union)
        /
        (c_area + 1e-6)
    )

    return giou

class DETRLoss(nn.Module):

    def __init__(
        self,
        num_classes=2
    ):
        super().__init__()

        self.matcher = HungarianMatcher()

        self.num_classes = num_classes

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1
        self.register_buffer("empty_weight", empty_weight)

        self.l1_loss = nn.L1Loss()

        self.giou_weight = 2

    def forward(
        self,
        outputs,
        targets
    ):

        indices = self.matcher(
            outputs,
            targets
        )

        total_cls_loss = 0
        total_bbox_loss = 0

        bs = len(indices)

        for b, (pred_idx, tgt_idx) in enumerate(indices):

            pred_logits = outputs[
                "pred_logits"
            ][b]

            pred_boxes = outputs[
                "pred_boxes"
            ][b]

            target_labels = targets[
                b
            ]["class_labels"]

            target_boxes = targets[
                b
            ]["boxes"]

            matched_logits = pred_logits[
                pred_idx
            ]

            matched_boxes = pred_boxes[
                pred_idx
            ]

            matched_labels = target_labels[
                tgt_idx
            ]

            matched_target_boxes = target_boxes[
                tgt_idx
            ]

            # Create target classes filled with background class index (self.num_classes)
            target_classes = torch.full(
                (pred_logits.shape[0],),
                self.num_classes,
                dtype=torch.long,
                device=pred_logits.device
            )
            # Set target labels for matched queries
            target_classes[pred_idx] = matched_labels

            cls_loss = F.cross_entropy(
                pred_logits,
                target_classes,
                weight=self.empty_weight
            )

            if len(pred_idx) > 0:
                bbox_loss = self.l1_loss(
                    matched_boxes,
                    matched_target_boxes
                )

                pred_xyxy = box_cxcywh_to_xyxy(
                    matched_boxes
                )

                target_xyxy = box_cxcywh_to_xyxy(
                    matched_target_boxes
                )

                giou = generalized_box_iou(
                    pred_xyxy,
                    target_xyxy
                )

                giou_loss = (
                    1 - giou
                ).mean()
            else:
                bbox_loss = torch.tensor(0.0, device=pred_boxes.device)
                giou_loss = torch.tensor(0.0, device=pred_boxes.device)

            total_cls_loss += cls_loss

            total_bbox_loss += (
                bbox_loss
                +
                self.giou_weight * giou_loss
            )

        total_loss = (
            total_cls_loss
            +
            5 * total_bbox_loss
        )

        return {
            "loss": total_loss,
            "cls_loss": total_cls_loss,
            "bbox_loss": total_bbox_loss
        }