"""
train.py — Master training script for ForgeryNet.

Supports:
- Stage 1: Baseline CNN (Downsampled 512x512, Unweighted BCE + Dice)
- Stage 2: Balanced Loss (Downsampled 512x512, Focal Tversky + Balanced BCE)
- Stage 3: Dual-Stream + Spatial Attention (Downsampled 512x512)
- Stage 4: Native-Resolution Patch Pipeline (Final Proposed SOTA)
"""

import argparse
import os
import sys
import time
import csv
import shutil

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import make_dataloaders
from src.model import ForgeryNet
from src.losses import MultiTaskLoss
from src.utils import seed_everything
from src.evaluate import evaluate_model


def parse_args():
    p = argparse.ArgumentParser(description="Train ForgeryNet across all developmental stages")
    p.add_argument("--stage", type=int, default=4, choices=[1, 2, 3, 4],
                   help="1: Baseline, 2: Balanced Loss, 3: Dual-Stream + Attention, 4: Native Patch (SOTA)")
    p.add_argument("--data_dir", type=str, default="Bengali_Handwriting_CopyMove_Dataset")
    p.add_argument("--out_dir",  type=str, default="outputs")
    p.add_argument("--epochs",   type=int, default=35)
    p.add_argument("--bs",       type=int, default=8, help="Batch size")
    p.add_argument("--lr",       type=float, default=1.5e-4, help="Learning rate")
    p.add_argument("--wd",       type=float, default=1e-4, help="Weight decay")
    p.add_argument("--workers",  type=int, default=0, help="DataLoader workers")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"       LAUNCHING FORGERYNET TRAINING: STAGE {args.stage}")
    print("=" * 65)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Configure stage settings
    if args.stage == 1:
        print("Stage 1 Configuration: Baseline CNN (Downsampled, Unweighted Losses)")
        mode = "fullpage"
        dual_stream = False
        use_attention = False
        criterion = MultiTaskLoss(lambda_det=1.0, lambda_loc=2.0, lambda_type=1.0, use_focal_tversky=False, pos_weight_loc=1.0, pos_weight_det=None)
    elif args.stage == 2:
        print("Stage 2 Configuration: Balanced Loss (Downsampled, Focal Tversky + pos_weight)")
        mode = "fullpage"
        dual_stream = False
        use_attention = False
        criterion = MultiTaskLoss(lambda_det=1.0, lambda_loc=4.0, lambda_type=1.0, use_focal_tversky=True, pos_weight_loc=15.0, pos_weight_det=0.33)
    elif args.stage == 3:
        print("Stage 3 Configuration: Dual-Stream Forensics + Spatial Attention (Downsampled)")
        mode = "fullpage"
        dual_stream = True
        use_attention = True
        criterion = MultiTaskLoss(lambda_det=1.0, lambda_loc=4.0, lambda_type=1.0, use_focal_tversky=True, pos_weight_loc=15.0, pos_weight_det=0.33)
    else:
        print("Stage 4 Configuration: Native-Resolution Patch Pipeline (FINAL PROPOSED SOTA)")
        mode = "patch"
        dual_stream = True
        use_attention = True
        criterion = MultiTaskLoss(lambda_det=1.0, lambda_loc=4.0, lambda_type=1.0, use_focal_tversky=True, pos_weight_loc=15.0, pos_weight_det=0.33)

    # Dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(
        root_dir=args.data_dir,
        mode=mode,
        batch_size=args.bs,
        num_workers=args.workers,
        seed=args.seed
    )

    # Initialize model
    model = ForgeryNet(
        pretrained=True,
        dual_stream=dual_stream,
        use_attention=use_attention
    ).to(device)

    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    out_stage_dir = os.path.join(args.out_dir, f"stage_{args.stage}")
    ckpt_dir = os.path.join(out_stage_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(out_stage_dir, "train_log.csv")

    fields = ["epoch", "lr", "train_total", "train_detect", "train_localize", "train_type",
              "val_total", "val_detect", "val_localize", "val_type", "epoch_time_s"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_val_loss = float("inf")
    patience_cnt = 0

    print(f"Beginning training for {args.epochs} epochs. Output: {out_stage_dir}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot_tr = {"total": 0.0, "detect": 0.0, "localize": 0.0, "type": 0.0}
        n_tr = 0

        for imgs, bl, tl, masks in train_loader:
            imgs, bl, tl, masks = imgs.to(device), bl.to(device), tl.to(device), masks.to(device)
            optimiser.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                d, s, t = model(imgs)
                losses = criterion(d, s, t, bl, tl, masks)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()

            for k in tot_tr:
                tot_tr[k] += losses[k].item()
            n_tr += 1

        scheduler.step()

        # Validation
        model.eval()
        tot_vl = {"total": 0.0, "detect": 0.0, "localize": 0.0, "type": 0.0}
        n_vl = 0
        with torch.no_grad():
            for imgs, bl, tl, masks in val_loader:
                imgs, bl, tl, masks = imgs.to(device), bl.to(device), tl.to(device), masks.to(device)
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    d, s, t = model(imgs)
                    losses = criterion(d, s, t, bl, tl, masks)
                for k in tot_vl:
                    tot_vl[k] += losses[k].item()
                n_vl += 1

        val_total = tot_vl["total"] / max(n_vl, 1)
        train_total = tot_tr["total"] / max(n_tr, 1)
        epoch_time = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | Train Loss: {train_total:.4f} | Val Loss: {val_total:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        row = {
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train_total": train_total, "train_detect": tot_tr["detect"] / n_tr,
            "train_localize": tot_tr["localize"] / n_tr, "train_type": tot_tr["type"] / n_tr,
            "val_total": val_total, "val_detect": tot_vl["detect"] / n_vl,
            "val_localize": tot_vl["localize"] / n_vl, "val_type": tot_vl["type"] / n_vl,
            "epoch_time_s": round(epoch_time, 2)
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_cnt = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_total},
                       os.path.join(ckpt_dir, "best.pth"))
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Loss: {val_total:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    print(f"\nTraining completed. Best checkpoint saved to: {os.path.join(ckpt_dir, 'best.pth')}")
    print("Running final evaluation on test set using best checkpoint...")
    best_ckpt = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
