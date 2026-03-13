---
name: SimCLR Chest X-ray Project
description: Contrastive learning medical image classification project using NIH Chest X-ray14 dataset
type: project
---

SimCLR self-supervised pre-training + supervised fine-tuning for NIH Chest X-ray14 multi-label classification.

**Why:** User requested a complete contrastive learning medical image classification project using a Kaggle public dataset.

**How to apply:** Reference when continuing this project in future sessions.

Key details:
- Dataset: NIH Chest X-ray14 (`nih-chest-xrays/data` on Kaggle), 112k chest X-rays, 15 classes (multi-label)
- Framework: SimCLR with NT-Xent loss, ResNet50 backbone adapted for grayscale (1-channel)
- Workflow: download → preprocess → pretrain → finetune → evaluate
- Three comparison modes: full_finetune, linear_probe, imagenet_baseline
- Apple Silicon MPS supported; set PYTORCH_ENABLE_MPS_FALLBACK=1
- Patient-level data splitting is critical to avoid leakage
