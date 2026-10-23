"""
train.py — Training loop for the ForgeryNet multi-task pipeline.

Usage
-----
    python src/train.py                         # default settings
    python src/train.py --epochs 50 --bs 8     # custom batch size
    python src/train.py --debug                 # 2-batch smoke test, no pretrained weights

Key features
------------
- AdamW optimiser + CosineAnnealingLR scheduler
- Mixed-precision (torch.cuda.amp) for faster training on CUDA
- Saves best checkpoint (lowest val total loss) and periodic checkpoints
- Per-epoch CSV log: all partial losses + quick metric snapshots
- Early stopping with configurable patience
"""

import argparse
import os
import sys
import time
import csv

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow importing from src/ when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import make_dataloaders
from src.model   import ForgeryNet
from src.losses  import MultiTaskLoss
from src.utils   import seed_everything

# Optional: import evaluate helpers for quick val-epoch metrics
from src.evaluate import _quick_metrics


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ForgeryNet")
    p.add_argument(
        "--data_dir", type=str,
        default=r"c:\Users\DEBJANI KAR\Desktop\Forgery Detection\Bengali_Handwriting_CopyMove_Dataset",
        help="Path to Bengali_Handwriting_CopyMove_Dataset/",
    )
    p.add_argument("--out_dir",   type=str, default="outputs",  help="Output directory")
    p.add_argument("--epochs",    type=int, default=50,         help="Number of training epochs")
    p.add_argument("--bs",        type=int, default=8,          help="Batch size")
    p.add_argument("--lr",        type=float, default=1e-4,     help="Initial learning rate")
    p.add_argument("--wd",        type=float, default=1e-4,     help="Weight decay")
    p.add_argument("--workers",   type=int, default=4,          help="DataLoader workers")
    p.add_argument("--img_size",  type=int, default=512,        help="Input image size (square)")
    p.add_argument("--seed",      type=int, default=42,         help="Random seed")
    p.add_argument("--patience",  type=int, default=10,         help="Early-stopping patience")
    p.add_argument("--ckpt_every",type=int, default=5,          help="Save checkpoint every N epochs")
    p.add_argument("--lambda_det",  type=float, default=1.0)
    p.add_argument("--lambda_loc",  type=float, default=2.0)
    p.add_argument("--lambda_type", type=float, default=1.0)
    p.add_argument("--no_pretrain", action="store_true",        help="Train from scratch")
    p.add_argument("--debug",       action="store_true",        help="Smoke-test: 2 batches, no pretrain")
    return p.parse_args()


# -----------------------------------------------------------------------------
# One training epoch
# -----------------------------------------------------------------------------

def train_one_epoch(
    model:     ForgeryNet,
    loader,
    criterion: MultiTaskLoss,
    optimiser: optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    debug:     bool = False,
) -> dict:
    model.train()
    totals = {"total": 0.0, "detect": 0.0, "localize": 0.0, "type": 0.0}
    n_batches = 0

    for batch_idx, (images, bin_lbl, type_lbl, masks) in enumerate(loader):
        if debug and batch_idx >= 2:
            break

        images   = images.to(device, non_blocking=True)
        bin_lbl  = bin_lbl.to(device, non_blocking=True)
        type_lbl = type_lbl.to(device, non_blocking=True)
        masks    = masks.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            det_logit, seg_mask, type_logit = model(images)
            losses = criterion(
                det_logit, seg_mask, type_logit,
                bin_lbl, type_lbl, masks,
            )

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimiser)
        scaler.update()

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# -----------------------------------------------------------------------------
# One validation epoch
# -----------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:     ForgeryNet,
    loader,
    criterion: MultiTaskLoss,
    device:    torch.device,
    debug:     bool = False,
) -> dict:
    model.eval()
    totals = {"total": 0.0, "detect": 0.0, "localize": 0.0, "type": 0.0}
    n_batches = 0

    # Accumulators for quick metrics
    all_det_pred, all_det_gt   = [], []
    all_type_pred, all_type_gt = [], []
    all_dice                   = []

    for batch_idx, (images, bin_lbl, type_lbl, masks) in enumerate(loader):
        if debug and batch_idx >= 2:
            break

        images   = images.to(device, non_blocking=True)
        bin_lbl  = bin_lbl.to(device, non_blocking=True)
        type_lbl = type_lbl.to(device, non_blocking=True)
        masks    = masks.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            det_logit, seg_mask, type_logit = model(images)
            losses = criterion(
                det_logit, seg_mask, type_logit,
                bin_lbl, type_lbl, masks,
            )

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

        # Collect predictions for metrics
        det_prob = det_logit.sigmoid().squeeze(1).cpu()
        all_det_pred.append(det_prob)
        all_det_gt.append(bin_lbl.cpu())

        # Type metrics — forged samples only
        forged_idx = (bin_lbl == 1).cpu()
        if forged_idx.any():
            tp = type_logit.argmax(dim=1).cpu()[forged_idx]
            tg = type_lbl.cpu()[forged_idx]
            all_type_pred.append(tp)
            all_type_gt.append(tg)

        # Dice per sample
        all_dice.append(_quick_metrics(seg_mask.cpu(), masks.cpu()))

    avg_losses = {k: v / max(n_batches, 1) for k, v in totals.items()}

    # Det F1
    det_f1 = _compute_f1(
        torch.cat(all_det_pred), torch.cat(all_det_gt), threshold=0.5
    )
    avg_dice = float(torch.cat(all_dice).mean()) if all_dice else 0.0

    # Type macro-F1 (forged only)
    if all_type_pred:
        type_f1 = _compute_macro_f1(
            torch.cat(all_type_pred), torch.cat(all_type_gt), num_classes=3
        )
    else:
        type_f1 = 0.0

    avg_losses["val_det_f1"]  = det_f1
    avg_losses["val_dice"]    = avg_dice
    avg_losses["val_type_f1"] = type_f1
    return avg_losses


# -----------------------------------------------------------------------------
# Inline metric helpers (lightweight, no sklearn needed at train-time)
# -----------------------------------------------------------------------------

def _compute_f1(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (probs > threshold).long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)


def _compute_macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(sum(f1s) / num_classes)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    debug  = args.debug

    if debug:
        print("[train] DEBUG mode - 2 batches per epoch, no pretrained weights")
        args.epochs      = 2
        args.no_pretrain = True
        args.patience    = 999

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # -- Directories -----------------------------------------------------------
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    log_path = os.path.join(args.out_dir, "train_log.csv")
    os.makedirs(ckpt_dir, exist_ok=True)

    # -- Data -----------------------------------------------------------------
    img_size = (args.img_size, args.img_size)
    train_loader, val_loader, _ = make_dataloaders(
        root_dir    = args.data_dir,
        batch_size  = args.bs,
        num_workers = 0 if debug else args.workers,
        seed        = args.seed,
        img_size    = img_size,
    )

    # -- Model -----------------------------------------------------------------
    model     = ForgeryNet(pretrained=not args.no_pretrain).to(device)
    criterion = MultiTaskLoss(
        lambda_det  = args.lambda_det,
        lambda_loc  = args.lambda_loc,
        lambda_type = args.lambda_type,
    )
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    # -- CSV logger ------------------------------------------------------------
    log_fields = [
        "epoch", "lr",
        "train_total", "train_detect", "train_localize", "train_type",
        "val_total",   "val_detect",   "val_localize",   "val_type",
        "val_det_f1",  "val_dice",     "val_type_f1",
        "epoch_time_s",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # -- Training loop ---------------------------------------------------------
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, criterion, optimiser, scaler, device, debug)
        val_m   = validate(model, val_loader, criterion, device, debug)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # -- Logging ----------------------------------------------------------
        row = {
            "epoch":          epoch,
            "lr":             f"{current_lr:.2e}",
            "train_total":    f"{train_m['total']:.4f}",
            "train_detect":   f"{train_m['detect']:.4f}",
            "train_localize": f"{train_m['localize']:.4f}",
            "train_type":     f"{train_m['type']:.4f}",
            "val_total":      f"{val_m['total']:.4f}",
            "val_detect":     f"{val_m['detect']:.4f}",
            "val_localize":   f"{val_m['localize']:.4f}",
            "val_type":       f"{val_m['type']:.4f}",
            "val_det_f1":     f"{val_m['val_det_f1']:.4f}",
            "val_dice":       f"{val_m['val_dice']:.4f}",
            "val_type_f1":    f"{val_m['val_type_f1']:.4f}",
            "epoch_time_s":   f"{elapsed:.1f}",
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}]  "
            f"train={train_m['total']:.4f}  val={val_m['total']:.4f}  "
            f"det-F1={val_m['val_det_f1']:.3f}  "
            f"dice={val_m['val_dice']:.3f}  "
            f"type-F1={val_m['val_type_f1']:.3f}  "
            f"lr={current_lr:.2e}  {elapsed:.0f}s"
        )

        # -- Checkpointing -----------------------------------------------------
        is_best = val_m["total"] < best_val_loss
        if is_best:
            best_val_loss = val_m["total"]
            no_improve    = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimiser": optimiser.state_dict(), "val_loss": best_val_loss},
                os.path.join(ckpt_dir, "best.pth"),
            )
            print(f"  [BEST] Checkpoint saved  (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1

        if epoch % args.ckpt_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict()},
                os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"),
            )

        # -- Early stopping ----------------------------------------------------
        if no_improve >= args.patience:
            print(f"[train] Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {args.patience} epochs).")
            break

    print(f"\n[train] Training complete. Logs -> {log_path}")
    print(f"[train] Best val loss: {best_val_loss:.4f}")

    # -- Plot training curves --------------------------------------------------
    try:
        from src.utils import plot_training_curves
        plot_training_curves(log_path, save_dir=args.out_dir)
    except Exception as e:
        print(f"[train] Could not plot curves: {e}")


if __name__ == "__main__":
    main()
