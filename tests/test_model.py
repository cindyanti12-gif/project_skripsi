import torch

from model.detr_fusion import DETRVGGFusion

model = DETRVGGFusion()

x = torch.randn(
    1,
    3,
    800,
    800
)

outputs = model(x)

print("Logits Shape:")
print(outputs["pred_logits"].shape)

print()

print("Boxes Shape:")
print(outputs["pred_boxes"].shape)