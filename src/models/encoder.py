"""
Backbone encoder adapted for grayscale chest X-ray images.

Supports multiple architectures via a registry:
  - ResNet family   : resnet18, resnet34, resnet50, resnet101
  - EfficientNet    : efficientnet_b0, efficientnet_b1, efficientnet_b2
  - Vision Transformer : vit_b_16, vit_b_32, vit_l_16

Key modifications vs. standard models:
  1. First conv / patch embedding: 3 input channels → 1 (grayscale).
  2. Classification head is removed; only the feature extractor is retained.
  3. Output: a flat feature vector whose dimension depends on the backbone.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

_BACKBONE_REGISTRY: dict[str, dict] = {}


def _register(name: str, *, constructor, weights, feature_dim: int, family: str):
    _BACKBONE_REGISTRY[name] = {
        "constructor": constructor,
        "weights": weights,
        "feature_dim": feature_dim,
        "family": family,
    }


# ResNet variants
_register("resnet18",  constructor=models.resnet18,  weights=models.ResNet18_Weights.IMAGENET1K_V1,  feature_dim=512,  family="resnet")
_register("resnet34",  constructor=models.resnet34,  weights=models.ResNet34_Weights.IMAGENET1K_V1,  feature_dim=512,  family="resnet")
_register("resnet50",  constructor=models.resnet50,  weights=models.ResNet50_Weights.IMAGENET1K_V1,  feature_dim=2048, family="resnet")
_register("resnet101", constructor=models.resnet101, weights=models.ResNet101_Weights.IMAGENET1K_V1, feature_dim=2048, family="resnet")

# EfficientNet variants
_register("efficientnet_b0", constructor=models.efficientnet_b0, weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1, feature_dim=1280, family="efficientnet")
_register("efficientnet_b1", constructor=models.efficientnet_b1, weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1, feature_dim=1280, family="efficientnet")
_register("efficientnet_b2", constructor=models.efficientnet_b2, weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1, feature_dim=1408, family="efficientnet")

# Vision Transformer variants
_register("vit_b_16", constructor=models.vit_b_16, weights=models.ViT_B_16_Weights.IMAGENET1K_V1, feature_dim=768, family="vit")
_register("vit_b_32", constructor=models.vit_b_32, weights=models.ViT_B_32_Weights.IMAGENET1K_V1, feature_dim=768, family="vit")
_register("vit_l_16", constructor=models.vit_l_16, weights=models.ViT_L_16_Weights.IMAGENET1K_V1, feature_dim=1024, family="vit")


def available_backbones() -> list[str]:
    """Return list of supported backbone names."""
    return list(_BACKBONE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Per-family backbone builders
# ---------------------------------------------------------------------------

def _build_resnet(entry: dict, pretrained_imagenet: bool) -> tuple[nn.Module, int]:
    weights = entry["weights"] if pretrained_imagenet else None
    resnet = entry["constructor"](weights=weights)

    # Adapt first conv for single-channel grayscale input
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
        resnet.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

    # Strip avgpool + FC, keep feature extractor
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    return backbone, entry["feature_dim"]


def _build_efficientnet(entry: dict, pretrained_imagenet: bool) -> tuple[nn.Module, int]:
    weights = entry["weights"] if pretrained_imagenet else None
    effnet = entry["constructor"](weights=weights)

    # Adapt first conv for grayscale
    original_conv = effnet.features[0][0]
    effnet.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )
    if pretrained_imagenet:
        effnet.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

    # features + avgpool, drop classifier
    backbone = nn.Sequential(effnet.features, effnet.avgpool)
    return backbone, entry["feature_dim"]


def _build_vit(entry: dict, pretrained_imagenet: bool) -> tuple[nn.Module, int]:
    weights = entry["weights"] if pretrained_imagenet else None
    vit = entry["constructor"](weights=weights)

    # Adapt patch embedding conv for grayscale
    original_conv = vit.conv_proj
    vit.conv_proj = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
    )
    if pretrained_imagenet:
        vit.conv_proj.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        if original_conv.bias is not None:
            vit.conv_proj.bias.data = original_conv.bias.data.clone()

    # Remove classification head — we wrap in a helper that returns the CLS token
    vit.heads = nn.Identity()
    return vit, entry["feature_dim"]


_FAMILY_BUILDERS = {
    "resnet": _build_resnet,
    "efficientnet": _build_efficientnet,
    "vit": _build_vit,
}


# ---------------------------------------------------------------------------
# GradCAM target layer helpers
# ---------------------------------------------------------------------------

def get_gradcam_target_layer(encoder: "SimCLREncoder") -> list[nn.Module]:
    """Return the appropriate GradCAM target layer(s) for the encoder's backbone family."""
    family = encoder.family
    if family == "resnet":
        # Last block of the last ResNet layer (layer4)
        return [encoder.backbone[-2][-1]]
    elif family == "efficientnet":
        # Last conv block in features
        return [encoder.backbone[0][-1]]
    elif family == "vit":
        # Last encoder block
        return [encoder.backbone.encoder.layers[-1]]
    else:
        raise ValueError(f"GradCAM not supported for backbone family: {family}")


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SimCLREncoder(nn.Module):
    """
    Feature extractor supporting multiple backbone architectures.

    All backbones are adapted for single-channel grayscale input and strip
    their classification heads, outputting a flat feature vector.

    Args:
        backbone            : model name (see available_backbones() for options).
        pretrained_imagenet : load ImageNet-1k weights (useful for baseline comparison).
    """

    def __init__(self, backbone: str = "resnet50", pretrained_imagenet: bool = False):
        super().__init__()

        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unsupported backbone: '{backbone}'. "
                f"Available: {available_backbones()}"
            )

        entry = _BACKBONE_REGISTRY[backbone]
        self.family = entry["family"]
        builder = _FAMILY_BUILDERS[self.family]
        self.backbone, self.feature_dim = builder(entry, pretrained_imagenet)

    def forward(self, x):
        """x: (B, 1, H, W) -> h: (B, feature_dim)"""
        h = self.backbone(x)
        return h.flatten(1)
