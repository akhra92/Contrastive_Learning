"""
ResNet50 backbone adapted for grayscale chest X-ray images.

Key modifications vs. standard ResNet50:
  1. First conv: 3 input channels → 1 (grayscale).
  2. Average-pooling head and FC classifier are removed;
     only the feature extractor is retained.
  3. Output: 2048-dim feature vector (before projection head).
"""

import torch.nn as nn
import torchvision.models as models


class SimCLREncoder(nn.Module):
    """
    ResNet50 feature extractor for SimCLR pre-training and fine-tuning.

    Args:
        backbone            : model name (only 'resnet50' currently supported).
        pretrained_imagenet : load ImageNet-1k weights (useful for baseline comparison).
    """

    def __init__(self, backbone: str = "resnet50", pretrained_imagenet: bool = False):
        super().__init__()

        if backbone != "resnet50":
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet50'.")

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
        resnet = models.resnet50(weights=weights)

        # Adapt first convolution for single-channel grayscale input
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        if pretrained_imagenet:
            # Average ImageNet RGB weights across channels to initialise the
            # single-channel conv with meaningful values.
            resnet.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        # Strip the final average-pool + FC (keep layers 0-8, drop avgpool/fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 2048, 1, 1)
        self.feature_dim = 2048

    def forward(self, x):
        """x: (B, 1, H, W) -> h: (B, 2048)"""
        h = self.backbone(x)
        return h.flatten(1)
