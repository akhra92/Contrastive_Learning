"""
SimCLR self-supervised pre-training loop for NIH Chest X-ray14.

Trains a ResNet50 encoder + projection head using the NT-Xent contrastive loss.
At the end of each checkpoint interval, ONLY the encoder weights are saved —
the projection head is intentionally discarded after pre-training (SimCLR paper
§3: "We use the representation before the non-linear projection head for downstream tasks").
"""

import os
import glob
import time
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.augmentations import SimCLRAugmentation
from src.data.dataset import SimCLRDataset
from src.losses.nt_xent import NTXentLoss
from src.models.encoder import SimCLREncoder
from src.models.projection_head import ProjectionHead
from src.training.utils import (
    AverageMeter,
    cosine_schedule_with_warmup,
    get_device,
    save_checkpoint,
)


def _collect_image_paths(raw_dir: str) -> list:
    """Recursively collect all .png image paths for pre-training."""
    paths = glob.glob(os.path.join(raw_dir, "**", "*.png"), recursive=True)
    if not paths:
        raise FileNotFoundError(
            f"No .png images found under {raw_dir}. "
            "Run scripts/download_data.sh first."
        )
    return paths


def pretrain(config: dict):
    train_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    ckpt_cfg = config["checkpointing"]
    log_cfg = config["logging"]

    device = get_device(train_cfg["device"])
    print(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Dataset & DataLoader                                                 #
    # ------------------------------------------------------------------ #
    image_paths = _collect_image_paths(data_cfg["raw_dir"])
    print(f"Pre-training on {len(image_paths)} images")

    augmentation = SimCLRAugmentation(config)
    dataset = SimCLRDataset(image_paths, augmentation)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"] and device.type != "mps",
        drop_last=True,  # NT-Xent requires consistent batch size
    )

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    encoder = SimCLREncoder(
        backbone=model_cfg["backbone"],
        pretrained_imagenet=model_cfg["pretrained_imagenet"],
    ).to(device)

    proj_head = ProjectionHead(
        input_dim=encoder.feature_dim,
        hidden_dim=model_cfg["projection_hidden_dim"],
        output_dim=model_cfg["projection_dim"],
    ).to(device)

    criterion = NTXentLoss(temperature=train_cfg["temperature"])

    # ------------------------------------------------------------------ #
    # Optimiser & scheduler                                                #
    # ------------------------------------------------------------------ #
    params = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.Adam(
        params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=train_cfg["warmup_epochs"],
        total_epochs=train_cfg["epochs"],
    )

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    os.makedirs(ckpt_cfg["save_dir"], exist_ok=True)
    os.makedirs(log_cfg["log_dir"], exist_ok=True)

    loss_history = []
    best_loss = float("inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        encoder.train()
        proj_head.train()
        meter = AverageMeter("loss")
        t0 = time.time()

        for step, (view1, view2) in enumerate(loader, start=1):
            view1 = view1.to(device)
            view2 = view2.to(device)

            h1 = encoder(view1)
            h2 = encoder(view2)
            z1 = proj_head(h1)
            z2 = proj_head(h2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter.update(loss.item(), n=view1.size(0))

            if step % log_cfg["log_every"] == 0:
                print(
                    f"  Epoch {epoch:03d} | Step {step:04d}/{len(loader):04d}"
                    f" | Loss {meter.avg:.4f}"
                    f" | LR {optimizer.param_groups[0]['lr']:.2e}"
                )

        scheduler.step()
        epoch_loss = meter.avg
        loss_history.append(epoch_loss)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{train_cfg['epochs']:03d}"
            f" | Loss {epoch_loss:.4f}"
            f" | LR {optimizer.param_groups[0]['lr']:.2e}"
            f" | {elapsed:.1f}s"
        )

        # ---- Checkpointing ------------------------------------------- #
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        if epoch % ckpt_cfg["save_every"] == 0 or is_best:
            ckpt_path = os.path.join(ckpt_cfg["save_dir"], f"encoder_ep{epoch:03d}.pth")
            save_checkpoint(
                {"encoder": encoder.state_dict(), "epoch": epoch, "loss": epoch_loss},
                ckpt_path,
            )
            if is_best:
                best_path = os.path.join(ckpt_cfg["save_dir"], "best_encoder.pth")
                save_checkpoint(
                    {"encoder": encoder.state_dict(), "epoch": epoch, "loss": epoch_loss},
                    best_path,
                )
                print(f"  [Best] Saved encoder to {best_path}")

    # Save final loss history
    log_path = os.path.join(log_cfg["log_dir"], "pretrain_loss.txt")
    with open(log_path, "w") as f:
        for ep, l in enumerate(loss_history, start=1):
            f.write(f"{ep}\t{l:.6f}\n")

    print(f"\nPre-training complete. Best loss: {best_loss:.4f}")
    print(f"Encoder checkpoint: {os.path.join(ckpt_cfg['save_dir'], 'best_encoder.pth')}")
