import torch

from loss.detr_loss import DETRLoss

criterion = DETRLoss()

outputs = {
    "pred_logits": torch.rand(
        2,
        25,
        3
    ),
    "pred_boxes": torch.rand(
        2,
        25,
        4
    )
}

targets = [
    {
        "class_labels": torch.tensor([1,0]),
        "boxes": torch.rand(2,4)
    },
    {
        "class_labels": torch.tensor([1]),
        "boxes": torch.rand(1,4)
    }
]

loss_dict = criterion(
    outputs,
    targets
)

print(loss_dict)