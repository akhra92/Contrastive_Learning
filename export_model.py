"""
Export a fine-tuned ChestXrayClassifier to ONNX and/or TorchScript formats.

Usage:
    python export_model.py --checkpoint checkpoints/finetune/best_model_full_finetune.pth
    python export_model.py --checkpoint <path> --format onnx torchscript
    python export_model.py --checkpoint <path> --format onnx --output_dir exports
    python export_model.py --checkpoint <path> --image_size 256
"""

import argparse
import os

import torch
import yaml

from src.models.classifier import ChestXrayClassifier
from src.models.encoder import SimCLREncoder
from src.training.utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Export model to ONNX / TorchScript")
    p.add_argument("--config", default="configs/finetune_config.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to fine-tuned model checkpoint")
    p.add_argument(
        "--format",
        nargs="+",
        choices=["onnx", "torchscript"],
        default=["onnx", "torchscript"],
        help="Export format(s) (default: both)",
    )
    p.add_argument("--output_dir", default="exports", help="Directory for exported models")
    p.add_argument("--image_size", type=int, default=None, help="Input image size (default: from config)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="cpu")
    return p.parse_args()


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> ChestXrayClassifier:
    model_cfg = config["model"]
    encoder = SimCLREncoder(backbone=model_cfg["backbone"])
    model = ChestXrayClassifier(
        encoder=encoder,
        num_classes=model_cfg["num_classes"],
        hidden_dim=model_cfg["classifier_hidden_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(
    model: ChestXrayClassifier,
    dummy_input: torch.Tensor,
    output_path: str,
    opset_version: int = 17,
):
    """Export model to ONNX format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX exported to {output_path} ({size_mb:.1f} MB)")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX model verification passed.")
    except ImportError:
        print("  Install 'onnx' package for model verification: pip install onnx")
    except Exception as e:
        print(f"  ONNX verification warning: {e}")


def export_torchscript(
    model: ChestXrayClassifier,
    dummy_input: torch.Tensor,
    output_path: str,
):
    """Export model to TorchScript format via tracing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    traced = torch.jit.trace(model, dummy_input)

    # Verify traced model produces same output
    with torch.no_grad():
        original_out = model(dummy_input)
        traced_out = traced(dummy_input)
        max_diff = (original_out - traced_out).abs().max().item()

    traced.save(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"TorchScript exported to {output_path} ({size_mb:.1f} MB)")
    print(f"  Verification max diff: {max_diff:.2e}")


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(args.device)
    # Export on CPU for maximum compatibility
    if device.type == "mps":
        device = torch.device("cpu")

    image_size = args.image_size or config["data"]["image_size"]
    model_cfg = config["model"]

    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Backbone    : {model_cfg['backbone']}")
    print(f"Image size  : {image_size}")
    print(f"Format(s)   : {', '.join(args.format)}")
    print(f"Output dir  : {args.output_dir}")
    print()

    model = load_model(config, args.checkpoint, device)
    dummy_input = torch.randn(1, 1, image_size, image_size, device=device)

    # Derive a base name from the checkpoint filename
    base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    if "onnx" in args.format:
        onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
        export_onnx(model, dummy_input, onnx_path, opset_version=args.opset)

    if "torchscript" in args.format:
        ts_path = os.path.join(args.output_dir, f"{base_name}.pt")
        export_torchscript(model, dummy_input, ts_path)

    print(f"\nExport complete. Files saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
