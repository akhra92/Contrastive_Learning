"""
Entry point for supervised fine-tuning / linear probe.

Three modes (set via --mode or config):
  full_finetune      — SimCLR encoder + end-to-end fine-tuning
  linear_probe       — SimCLR encoder frozen, classifier only
  imagenet_baseline  — ImageNet-pretrained ResNet50 baseline

Usage:
    python train_finetune.py
    python train_finetune.py --mode full_finetune
    python train_finetune.py --mode linear_probe --epochs 30
    python train_finetune.py --mode imagenet_baseline
    python train_finetune.py --checkpoint checkpoints/pretrain/encoder_ep100.pth
    python train_finetune.py --resume checkpoints/finetune/latest_finetune_full_finetune.pth
"""

import argparse
import yaml

from src.training.finetune import finetune


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune classifier on NIH Chest X-ray14")
    p.add_argument("--config", default="configs/finetune_config.yaml")
    p.add_argument(
        "--mode",
        choices=["full_finetune", "linear_probe", "imagenet_baseline"],
        help="Fine-tuning mode (overrides config)",
    )
    p.add_argument("--checkpoint", help="Path to pre-trained encoder checkpoint")
    p.add_argument("--epochs", type=int, help="Override training epochs")
    p.add_argument("--batch_size", type=int, help="Override batch size")
    p.add_argument("--lr", type=float, help="Override learning rate")
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], help="Override device")
    p.add_argument("--seed", type=int, help="Override random seed (default: 42)")
    p.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode:
        config["training"]["mode"] = args.mode
    if args.checkpoint:
        config["model"]["pretrained_checkpoint"] = args.checkpoint
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.device:
        config["training"]["device"] = args.device
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.wandb:
        config.setdefault("wandb", {})["enabled"] = True

    mode = config["training"]["mode"]
    print("=" * 60)
    print("Supervised Fine-tuning")
    print(f"  Config     : {args.config}")
    print(f"  Mode       : {mode}")
    print(f"  Backbone   : {config['model']['backbone']}")
    print(f"  Epochs     : {config['training']['epochs']}")
    print(f"  Batch size : {config['training']['batch_size']}")
    print(f"  LR         : {config['training']['lr']}")
    print(f"  Seed       : {config['training'].get('seed', 42)}")
    if mode != "imagenet_baseline":
        print(f"  Checkpoint : {config['model']['pretrained_checkpoint']}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print("=" * 60)

    finetune(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
