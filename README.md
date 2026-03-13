# Contrastive Learning for Medical Image Classification

Self-supervised pre-training with **SimCLR** on the **NIH Chest X-ray14** dataset, followed by supervised fine-tuning for multi-label pathology classification.

---

## Overview

This project demonstrates how contrastive self-supervised learning (SSL) can learn rich medical image representations **without labels**, which are then fine-tuned for downstream classification. The key advantage: SimCLR pre-training uses all 112k images (labels not needed), while supervised methods are limited to labelled training data.

### Pipeline

```
Raw X-ray images
      │
      ▼
[SimCLR Pre-training]          ← self-supervised, no labels used
  ResNet50 encoder
  + Projection head (MLP)
  + NT-Xent contrastive loss
      │
      ▼  (projection head discarded)
[Fine-tuning]                  ← supervised, multi-label BCE loss
  Pre-trained ResNet50 encoder
  + Classification head (MLP)
      │
      ▼
[Evaluation]
  Per-class AUC-ROC, ROC curves, t-SNE, GradCAM
```

### Three comparison modes

| Mode | Encoder init | Backbone frozen? |
|---|---|---|
| `full_finetune` | SimCLR pre-trained | No (end-to-end) |
| `linear_probe` | SimCLR pre-trained | Yes (classifier only) |
| `imagenet_baseline` | ImageNet weights | No |

---

## Dataset

**NIH Chest X-ray14** — 112,120 frontal-view chest X-rays from 30,805 patients.
Source: [kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)

**15 labels** (multi-label — one image can have multiple findings):

| Label | Label | Label |
|---|---|---|
| No Finding | Atelectasis | Cardiomegaly |
| Effusion | Infiltration | Mass |
| Nodule | Pneumonia | Pneumothorax |
| Consolidation | Edema | Emphysema |
| Fibrosis | Pleural Thickening | Hernia |

---

## Project Structure

```
Contrastive_Learning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── pretrain_config.yaml      # SimCLR hyperparameters
│   ├── finetune_config.yaml      # Fine-tuning hyperparameters
│   └── data_config.yaml          # Dataset paths and class names
│
├── scripts/
│   ├── download_data.sh          # Kaggle API download + preprocessing
│   ├── run_pretrain.sh           # Launch pre-training
│   ├── run_finetune.sh           # Launch fine-tuning
│   └── run_eval.sh               # Launch evaluation
│
├── src/
│   ├── data/
│   │   ├── augmentations.py      # SimCLR + finetune augmentation pipelines
│   │   ├── dataset.py            # SimCLRDataset, ChestXrayDataset
│   │   └── preprocess.py        # Build patient-level train/val/test splits
│   │
│   ├── models/
│   │   ├── encoder.py            # ResNet50 adapted for grayscale input
│   │   ├── projection_head.py    # 2-layer MLP projection head (SimCLR)
│   │   └── classifier.py        # Multi-label classification head
│   │
│   ├── losses/
│   │   └── nt_xent.py           # NT-Xent contrastive loss
│   │
│   ├── training/
│   │   ├── pretrain.py          # SimCLR pre-training loop
│   │   ├── finetune.py          # Supervised fine-tuning loop
│   │   └── utils.py             # Device selection, checkpointing, LR schedules
│   │
│   └── evaluation/
│       ├── metrics.py           # AUC-ROC, Average Precision, F1
│       └── visualize.py         # t-SNE, ROC curves, GradCAM, loss curves
│
├── train_pretrain.py             # Entry point: SimCLR pre-training
├── train_finetune.py             # Entry point: fine-tuning / linear probe
├── evaluate.py                   # Entry point: test set evaluation + plots
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Class distribution, sample images
│   ├── 02_augmentation_preview.ipynb # Visualise augmentation pairs
│   └── 03_results_analysis.ipynb    # ROC curves, t-SNE, metrics table
│
├── data/
│   ├── raw/                      # Downloaded dataset (gitignored)
│   └── processed/                # Generated CSV splits (gitignored)
│
├── checkpoints/
│   ├── pretrain/                 # Saved encoder checkpoints
│   └── finetune/                 # Saved classifier checkpoints
│
└── logs/                         # Training logs and output figures
```

---

## Setup

### 1. Clone / navigate to the project

```bash
cd /path/to/Contrastive_Learning
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** PyTorch MPS backend is automatically detected and used. No extra steps needed.
> **CUDA GPU:** Install the appropriate `torch` version from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Set up Kaggle API credentials

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Move the downloaded `kaggle.json` to `~/.kaggle/`:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Accept the dataset terms at [kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data) (required before downloading)

---

## Running the Project

### Step 1 — Download data and build splits

```bash
bash scripts/download_data.sh
```

This will:
- Download the full NIH Chest X-ray14 dataset (~45 GB) to `data/raw/`
- Run `src/data/preprocess.py` to create patient-level `train.csv`, `val.csv`, `test.csv` in `data/processed/`

> **Patient-level splitting:** All images from the same patient are kept in the same split to prevent data leakage. The official NIH test list is used as the test split.

### Step 2 — SimCLR pre-training

```bash
bash scripts/run_pretrain.sh
```

Or with custom arguments:

```bash
python train_pretrain.py --epochs 100 --batch_size 256 --device auto
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/pretrain_config.yaml` | Config file path |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 256 | Batch size (larger = more negatives = better) |
| `--lr` | 3e-4 | Learning rate |
| `--temperature` | 0.07 | NT-Xent temperature τ |
| `--device` | auto | `auto` / `mps` / `cuda` / `cpu` |

Checkpoints are saved to `checkpoints/pretrain/`. The best encoder is saved as `best_encoder.pth`.

> **Note:** Pre-training uses ALL images (labels ignored), so even test-split images participate — this is valid because no labels are used.

### Step 3 — Fine-tune for classification

**Full fine-tuning (recommended):**
```bash
python train_finetune.py --mode full_finetune
```

**Linear probe (backbone frozen):**
```bash
python train_finetune.py --mode linear_probe
```

**ImageNet baseline (for comparison):**
```bash
python train_finetune.py --mode imagenet_baseline
```

Or via the shell script:
```bash
bash scripts/run_finetune.sh --mode full_finetune
```

| Argument | Default | Description |
|---|---|---|
| `--mode` | `full_finetune` | Training mode (see above) |
| `--checkpoint` | from config | Path to pre-trained encoder |
| `--epochs` | 50 | Fine-tuning epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-4 | Classifier learning rate |
| `--device` | auto | Device |

Best models saved to `checkpoints/finetune/best_model_<mode>.pth`.

### Step 4 — Evaluate on the test set

```bash
python evaluate.py --checkpoint checkpoints/finetune/best_model_full_finetune.pth
```

Or via the shell script:
```bash
bash scripts/run_eval.sh --checkpoint checkpoints/finetune/best_model_full_finetune.pth
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--no_tsne` | Skip t-SNE (slow for large datasets) |
| `--no_gradcam` | Skip GradCAM generation |
| `--output_dir` | Directory for output figures (default: `logs/`) |

**Outputs saved to `logs/`:**
- `metrics_<mode>.txt` — per-class and macro-averaged AUC-ROC, AP, F1
- `roc_curves.png` — per-class ROC curves
- `tsne.png` — t-SNE of encoder embeddings coloured by pathology
- `gradcam_pneumonia.png` — GradCAM saliency maps
- `loss_curves.png` — pre-training and fine-tuning loss curves

---

## Notebooks

Open Jupyter Lab and run in order:

```bash
jupyter lab notebooks/
```

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | Class distribution, sample X-rays, split verification, pixel statistics |
| `02_augmentation_preview.ipynb` | Side-by-side view of original vs. SimCLR augmented pairs |
| `03_results_analysis.ipynb` | Compare all modes in a metrics table, ROC curves, t-SNE, GradCAM |

---

## Architecture Details

### SimCLR Encoder (ResNet50)

Standard ResNet50 with two modifications:
- **First conv:** `in_channels=3` → `in_channels=1` (grayscale X-rays)
- **Head removed:** Average-pooling + FC classifier stripped; outputs a 2048-dim feature vector

### Projection Head

2-layer MLP used only during pre-training, then discarded:
```
h (2048) → Linear → BN → ReLU → Linear → L2-normalise → z (128)
```

### NT-Xent Loss

For a batch of N images (→ 2N augmented views):

```
L = -log [ exp(sim(zᵢ, zⱼ) / τ) / Σ_{k≠i} exp(sim(zᵢ, zₖ) / τ) ]
```

where `sim` is cosine similarity and `τ = 0.07`. Each view's positive pair is the other augmented view of the same image; all other 2(N-1) views are negatives.

### Augmentation Pipeline (X-ray adapted)

SimCLR augmentations are tuned for grayscale medical images:
- Random resized crop (scale 0.08–1.0)
- Random horizontal flip
- Color jitter (brightness + contrast only — no saturation/hue for grayscale)
- Random Gaussian blur
- No `RandomGrayscale` (already grayscale)

### Classification Head

```
h (2048) → Linear(512) → BN → ReLU → Dropout(0.3) → Linear(15) → logits
```

- Loss: `BCEWithLogitsLoss` with per-class `pos_weight` for class imbalance
- Sigmoid applied at inference time (threshold = 0.5)
- Differential LR: backbone gets 10× lower LR than classifier head

---

## Expected Results

Macro AUC-ROC on NIH Chest X-ray14 test set with ResNet50:

| Method | Macro AUC-ROC |
|---|---|
| Random init + fine-tune | ~0.73 |
| ImageNet + fine-tune (baseline) | ~0.80 |
| SimCLR + linear probe | ~0.76 |
| **SimCLR + full fine-tune** | **~0.81–0.83** |

SimCLR pre-training on domain-specific data matches or exceeds ImageNet initialisation, demonstrating that self-supervised representations learn clinically meaningful features.

---

## Configuration

All hyperparameters are in YAML files under `configs/`. CLI arguments override config values.

**Key pre-training hyperparameters** (`configs/pretrain_config.yaml`):

```yaml
training:
  epochs: 100
  batch_size: 256      # critical: larger = more in-batch negatives
  temperature: 0.07    # NT-Xent τ — lower = sharper distribution
  lr: 3.0e-4
  warmup_epochs: 10    # linear LR warm-up before cosine decay
```

**Key fine-tuning hyperparameters** (`configs/finetune_config.yaml`):

```yaml
training:
  mode: "full_finetune"
  epochs: 50
  batch_size: 64
  lr: 1.0e-4
  backbone_lr_multiplier: 0.1   # backbone gets 10x lower LR
  use_class_weights: true       # compensate for class imbalance
```

---

## Troubleshooting

**`FileNotFoundError: No .png images found`**
→ Run `bash scripts/download_data.sh` first.

**`FileNotFoundError: data/processed/train.csv`**
→ Run `python src/data/preprocess.py` or re-run the download script.

**`FileNotFoundError: Checkpoint not found`**
→ Run pre-training first; then pass the correct path with `--checkpoint`.

**MPS out of memory**
→ Reduce `batch_size` in the config or CLI: `--batch_size 128`

**`ImportError: No module named 'pytorch_grad_cam'`**
→ GradCAM is optional. Skip it with `--no_gradcam`, or install: `pip install grad-cam`

**Slow pre-training on CPU**
→ Reduce `epochs` and `batch_size` for a quick test run:
`python train_pretrain.py --epochs 5 --batch_size 64`
