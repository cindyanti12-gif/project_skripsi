import torch
import torch.nn as nn
from torchvision import models


class VGGBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        vgg = models.vgg19(
            weights=models.VGG19_Weights.DEFAULT
        )

        self.features = vgg.features

        self.input_proj = nn.Conv2d(
            512,
            256,
            kernel_size=1
        )

    def forward(self, x):

        features = self.features(x)

        features = self.input_proj(features)

        return features