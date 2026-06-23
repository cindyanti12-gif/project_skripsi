import torch

from model.vgg_feature import VGGFeatureExtractor

model = VGGFeatureExtractor()

dummy = torch.randn(
    1,
    3,
    224,
    224
)

output = model(dummy)

print("Output Shape:")
print(output.shape)