"""
PyTorch Dataset classes for NIH Chest X-ray14.

Two distinct contracts:
  SimCLRDataset   — returns (view1, view2) augmented pairs; no labels.
                    Used for self-supervised pre-training.
  ChestXrayDataset — returns (image, label_vector) for supervised fine-tuning.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

ALL_CLASSES = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_CLASSES = len(ALL_CLASSES)


def build_label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert 'Finding Labels' column (pipe-separated) into a (N, 15) float32 matrix."""
    matrix = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    for i, findings in enumerate(df["Finding Labels"]):
        for label in str(findings).split("|"):
            label = label.strip()
            if label in ALL_CLASSES:
                matrix[i, ALL_CLASSES.index(label)] = 1.0
    return matrix


class SimCLRDataset(Dataset):
    """
    Unlabelled dataset for self-supervised pre-training.

    Accepts a list of absolute image paths and a SimCLRAugmentation callable.
    Labels are intentionally omitted so ALL available images (including the test
    split) can participate in pre-training without data leakage.
    """

    def __init__(self, image_paths: list, augmentation):
        self.image_paths = image_paths
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("L")  # grayscale
        view1, view2 = self.augmentation(image)
        return view1, view2


class ChestXrayDataset(Dataset):
    """
    Multi-label supervised dataset for NIH Chest X-ray14.

    Args:
        df         : DataFrame with columns ['Image Index', 'Finding Labels'].
        image_dir  : Directory containing the .png images.
        transform  : FinetuneAugmentation or any callable transform.
    """

    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.labels = build_label_matrix(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Image Index"])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
