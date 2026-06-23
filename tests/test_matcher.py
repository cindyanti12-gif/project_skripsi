import torch

from matcher import HungarianMatcher

matcher = HungarianMatcher()

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

indices = matcher(
    outputs,
    targets
)

print(indices)