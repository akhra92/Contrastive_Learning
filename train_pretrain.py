"""
Entry point for SimCLR self-supervised pre-training.

Usage:
    python train_pretrain.py
    python train_pretrain.py --config configs/pretrain_config.yaml
    python train_pretrain.py --epochs 200 --batch_size 512 --device mps
"""

import argparse
import yaml

from src.training.pretrain import pretrain


def parse_args():
    p = argparse.ArgumentParser(description="SimCLR pre-training on NIH Chest X-ray14")
    p.add_argument("--config", default="configs/pretrain_config.yaml")
    p.add_argument("--epochs", type=int, help="Override training epochs")
    p.add_argument("--batch_size", type=int, help="Override batch size")
    p.add_argument("--lr", type=float, help="Override learning rate")
    p.add_argument("--temperature", type=float, help="Override NT-Xent temperature")
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], help="Override device")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI arguments override config values
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.temperature:
        config["training"]["temperature"] = args.temperature
    if args.device:
        config["training"]["device"] = args.device
    if args.wandb:
        config.setdefault("wandb", {})["enabled"] = True

    print("=" * 60)
    print("SimCLR Pre-training")
    print(f"  Config          : {args.config}")
    print(f"  Backbone        : {config['model']['backbone']}")
    print(f"  Epochs          : {config['training']['epochs']}")
    print(f"  Batch size      : {config['training']['batch_size']}")
    print(f"  Temperature (τ) : {config['training']['temperature']}")
    print(f"  Learning rate   : {config['training']['lr']}")
    print(f"  Device          : {config['training']['device']}")
    print("=" * 60)

    pretrain(config)


if __name__ == "__main__":
    main()
