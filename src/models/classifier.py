"""
Classification head attached to the pre-trained SimCLR encoder for
supervised fine-tuning on NIH Chest X-ray14 (multi-label).

Supports three experimental modes:
  - full_finetune      : unfreeze all backbone layers, train end-to-end.
  - linear_probe       : freeze backbone, train only the classifier.
  - imagenet_baseline  : initialise backbone with ImageNet weights (no SimCLR).
"""

import torch.nn as nn

from src.models.encoder import SimCLREncoder


class ChestXrayClassifier(nn.Module):
    """
    Multi-label classifier wrapping a SimCLREncoder backbone.

    The classifier head is a 2-layer MLP with BatchNorm, ReLU, and Dropout.
    Raw logits are returned (no sigmoid); use BCEWithLogitsLoss for training
    and torch.sigmoid() at inference time.

    Args:
        encoder         : SimCLREncoder instance (pre-trained or random init).
        num_classes     : number of output labels (15 for NIH Chest X-ray14).
        freeze_backbone : if True, backbone parameters are frozen (linear probe).
        hidden_dim      : hidden dimension of the classification head.
        dropout         : dropout rate applied before the final linear layer.
    """

    def __init__(
        self,
        encoder: SimCLREncoder,
        num_classes: int = 15,
        freeze_backbone: bool = False,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(encoder.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """x: (B, 1, H, W) -> logits: (B, num_classes)"""
        h = self.encoder(x)
        return self.classifier(h)

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters (e.g., after linear probe warmup)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
