"""
Build train / val / test CSV splits from NIH Chest X-ray14 raw data.

Design decisions:
  - Uses the OFFICIAL test list (test_list.txt) provided by NIH to be comparable
    with published benchmarks.
  - Performs PATIENT-LEVEL splitting for train/val. Image-level splitting inflates
    metrics by 2-5 AUC points due to same-patient leakage across splits.
  - The test split is fixed; only train/val patient assignment uses randomness.

Usage:
    python src/data/preprocess.py
    python src/data/preprocess.py --raw_dir data/raw/nih-chest-xrays --out_dir data/processed
"""

import argparse
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def build_splits(raw_dir: str, processed_dir: str, val_fraction: float = 0.125, seed: int = 42):
    """
    Parameters
    ----------
    raw_dir       : path to the unzipped dataset folder.
    processed_dir : destination for train.csv, val.csv, test.csv.
    val_fraction  : fraction of train_val patients assigned to val
                    (0.125 ≈ 10 % of the full dataset).
    seed          : random seed for reproducibility.
    """
    entry_csv = os.path.join(raw_dir, "Data_Entry_2017.csv")
    train_val_txt = os.path.join(raw_dir, "train_val_list.txt")
    test_txt = os.path.join(raw_dir, "test_list.txt")

    if not os.path.isfile(entry_csv):
        raise FileNotFoundError(
            f"Data_Entry_2017.csv not found at {entry_csv}. "
            "Run scripts/download_data.sh first."
        )

    df = pd.read_csv(entry_csv)

    with open(train_val_txt) as f:
        train_val_files = set(f.read().splitlines())
    with open(test_txt) as f:
        test_files = set(f.read().splitlines())

    train_val_df = df[df["Image Index"].isin(train_val_files)].copy()
    test_df = df[df["Image Index"].isin(test_files)].copy()

    # Patient-level split to prevent leakage
    patient_ids = train_val_df["Patient ID"].unique()
    train_patients, val_patients = train_test_split(
        patient_ids, test_size=val_fraction, random_state=seed
    )

    train_df = train_val_df[train_val_df["Patient ID"].isin(train_patients)]
    val_df = train_val_df[train_val_df["Patient ID"].isin(val_patients)]

    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    print(f"Splits saved to {processed_dir}")
    print(f"  Train : {len(train_df):>6} images  ({train_df['Patient ID'].nunique()} patients)")
    print(f"  Val   : {len(val_df):>6} images  ({val_df['Patient ID'].nunique()} patients)")
    print(f"  Test  : {len(test_df):>6} images  ({test_df['Patient ID'].nunique()} patients)")
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Build dataset splits for NIH Chest X-ray14")
    parser.add_argument("--raw_dir", default="data/raw/nih-chest-xrays")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--val_fraction", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_splits(
        raw_dir=args.raw_dir,
        processed_dir=args.out_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
