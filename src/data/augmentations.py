"""
SimCLR augmentation pipeline adapted for chest X-rays (grayscale medical images).

Key differences from standard natural-image SimCLR:
  - Single-channel (L-mode) grayscale input — no saturation/hue jitter
  - Gaussian blur kernel size adjusted for 224x224 images
  - No RandomGrayscale (already grayscale)
  - Lighter crop scale preserved to keep pathological regions visible
"""

import random
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class SimCLRAugmentation:
    """Produce two independently augmented views of the same image for contrastive learning."""

    def __init__(self, config: dict):
        aug_cfg = config.get("augmentation", config)
        image_size = config.get("data", {}).get("image_size", config.get("image_size", 224))
        s = aug_cfg["color_jitter_strength"]

        # Kernel size must be odd and > 0; clamp to valid range
        k_min = aug_cfg.get("gaussian_blur_kernel_min", 3)
        k_max = aug_cfg.get("gaussian_blur_kernel_max", 23)
        # Pick a random odd kernel size at call time via a custom transform
        self._k_min = k_min if k_min % 2 == 1 else k_min + 1
        self._k_max = k_max if k_max % 2 == 1 else k_max - 1

        self._base_transforms = T.Compose([
            T.RandomResizedCrop(
                size=image_size,
                scale=tuple(aug_cfg["random_resized_crop_scale"]),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=aug_cfg["horizontal_flip_prob"]),
            T.RandomApply(
                [T.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0, hue=0)],
                p=aug_cfg["color_jitter_prob"],
            ),
            T.RandomApply(
                [_RandomGaussianBlur(k_min=self._k_min, k_max=self._k_max)],
                p=aug_cfg["gaussian_blur_prob"],
            ),
            T.ToTensor(),
            T.Normalize(mean=aug_cfg["normalize_mean"], std=aug_cfg["normalize_std"]),
        ])

    def __call__(self, image):
        """Return two independently augmented views of `image`."""
        return self._base_transforms(image), self._base_transforms(image)


class FinetuneAugmentation:
    """Standard augmentation for supervised fine-tuning."""

    def __init__(self, config: dict, train: bool = True):
        aug_cfg = config.get("augmentation", config)
        image_size = config.get("data", {}).get("image_size", config.get("image_size", 224))

        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(
                    size=image_size,
                    scale=tuple(aug_cfg["random_resized_crop_scale"]),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=aug_cfg["horizontal_flip_prob"]),
                T.ToTensor(),
                T.Normalize(mean=aug_cfg["normalize_mean"], std=aug_cfg["normalize_std"]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=aug_cfg["normalize_mean"], std=aug_cfg["normalize_std"]),
            ])

    def __call__(self, image):
        return self.transform(image)


class _RandomGaussianBlur:
    """Apply Gaussian blur with a randomly chosen odd kernel size."""

    def __init__(self, k_min: int = 3, k_max: int = 23):
        self.k_min = k_min
        self.k_max = k_max

    def __call__(self, img):
        k = random.randrange(self.k_min, self.k_max + 1, 2)  # odd values only
        return T.functional.gaussian_blur(img, kernel_size=k, sigma=(0.1, 2.0))
