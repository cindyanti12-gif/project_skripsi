import torch
from model.detr_fusion import VGGBackbone

model = VGGBackbone()

dummy = torch.randn(
    1,
    3,
    800,
    800
)

with torch.no_grad():
    output = model(dummy)

print("Output Shape:")
print(output.shape)