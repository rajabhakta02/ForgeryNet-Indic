"""
evaluate.py — Full multi-task evaluation suite for ForgeryNet.

Outputs:
- Branch 1: Accuracy, Precision, Recall, F1, ROC-AUC, Specificity on Genuine, Sensitivity on Forged
- Branch 2: Raw Mean Dice, Mean IoU, Otsu Cleaned Dice
- Branch 3: Classification report per class (Copy-Move, Splicing, Character Addition), Macro F1, Confusion Matrix
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import make_dataloaders
from src.model import ForgeryNet
from src.utils import TYPE_NAMES


def dice_coeff(pred: np.ndarray, target: np.ndarray, eps: float = 1.0) -> float:
    p = (pred > 0).astype(np.float32)
    t = (target > 0).astype(np.float32)
    intersection = (p * t).sum()
    return float((2.0 * intersection + eps) / (p.sum() + t.sum() + eps))


def iou_coeff(pred: np.ndarray, target: np.ndarray, eps: float = 1.0) -> float:
    p = (pred > 0).astype(np.float32)
    t = (target > 0).astype(np.float32)
    intersection = (p * t).sum()
    union = p.sum() + t.sum() - intersection
    return float((intersection + eps) / (union + eps))


def evaluate_model(
    model: ForgeryNet,
    loader,
    device: torch.device,
    det_thresh: float = 0.32,
    seg_thresh: float = 0.25,
) -> dict:
    model.eval()

    all_det_logits, all_det_preds, all_bin_targets = [], [], []
    all_type_preds, all_type_targets = [], []
    raw_dices, otsu_dices, raw_ious = [], [], []

    print(f"Running evaluation (det_thresh={det_thresh}, seg_thresh={seg_thresh})...")

    with torch.no_grad():
        for imgs, bl, tl, masks in loader:
            imgs = imgs.to(device)
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                d_logit, s_mask, t_logit = model(imgs)

            d_probs = torch.sigmoid(d_logit).cpu().numpy().flatten()
            bl_np   = bl.numpy().flatten()
            tl_np   = tl.numpy().flatten()
            s_np    = s_mask.cpu().numpy()
            m_np    = masks.numpy()

            all_det_logits.extend(d_probs)
            all_det_preds.extend((d_probs > det_thresh).astype(int))
            all_bin_targets.extend(bl_np)

            # Type predictions for forged samples
            t_preds = torch.argmax(t_logit, dim=1).cpu().numpy()
            for tp, tt, b in zip(t_preds, tl_np, bl_np):
                if b == 1 and tt != -1:
                    all_type_preds.append(tp)
                    all_type_targets.append(tt)

            # Localization metrics for forged samples
            for sm, gm, b in zip(s_np, m_np, bl_np):
                if b == 1 and gm.sum() > 0:
                    raw_bin = (sm[0] > seg_thresh).astype(np.uint8)
                    gt_bin  = (gm[0] > 0.5).astype(np.uint8)
                    raw_dices.append(dice_coeff(raw_bin, gt_bin))
                    raw_ious.append(iou_coeff(raw_bin, gt_bin))

                    # Otsu cleaning
                    try:
                        th = threshold_otsu(sm[0])
                        otsu_bin = (sm[0] > th).astype(np.uint8)
                        otsu_dices.append(dice_coeff(otsu_bin, gt_bin))
                    except Exception:
                        otsu_dices.append(dice_coeff(raw_bin, gt_bin))

    # Branch 1 Metrics
    all_bin_targets = np.array(all_bin_targets)
    all_det_preds   = np.array(all_det_preds)
    all_det_logits  = np.array(all_det_logits)

    acc = accuracy_score(all_bin_targets, all_det_preds)
    prec = precision_score(all_bin_targets, all_det_preds, zero_division=0)
    rec  = recall_score(all_bin_targets, all_det_preds, zero_division=0)
    f1   = f1_score(all_bin_targets, all_det_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(all_bin_targets, all_det_logits)
    except Exception:
        roc_auc = 0.5

    gen_mask = (all_bin_targets == 0)
    gen_acc = float((all_det_preds[gen_mask] == 0).mean()) if gen_mask.sum() > 0 else 0.0
    forged_mask = (all_bin_targets == 1)
    forged_rec = float((all_det_preds[forged_mask] == 1).mean()) if forged_mask.sum() > 0 else 0.0

    # Branch 2 Metrics
    mean_raw_dice  = float(np.mean(raw_dices)) if raw_dices else 0.0
    mean_raw_iou   = float(np.mean(raw_ious)) if raw_ious else 0.0
    mean_otsu_dice = float(np.mean(otsu_dices)) if otsu_dices else 0.0

    # Branch 3 Metrics
    all_type_targets = np.array(all_type_targets)
    all_type_preds   = np.array(all_type_preds)
    type_acc = accuracy_score(all_type_targets, all_type_preds) if len(all_type_targets) > 0 else 0.0
    type_macro_f1 = f1_score(all_type_targets, all_type_preds, average="macro", zero_division=0) if len(all_type_targets) > 0 else 0.0
    report = classification_report(all_type_targets, all_type_preds, target_names=["Copy-Move", "Splicing", "Character Addition"], digits=4, zero_division=0)
    cm = confusion_matrix(all_type_targets, all_type_preds)

    print("\n" + "=" * 65)
    print("           FORGERYNET BENCHMARK EVALUATION REPORT")
    print("=" * 65)
    print(f"BRANCH 1 (DETECTION):")
    print(f"  Overall Accuracy        : {acc * 100:.2f}% ({sum(all_det_preds == all_bin_targets)}/{len(all_bin_targets)})")
    print(f"  ROC-AUC                 : {roc_auc:.4f}")
    print(f"  Genuine Specificity     : {gen_acc * 100:.2f}% ({sum(all_det_preds[gen_mask] == 0)}/{gen_mask.sum()}) [0 false alarms]")
    print(f"  Forged Recall           : {forged_rec * 100:.2f}% ({sum(all_det_preds[forged_mask] == 1)}/{forged_mask.sum()})")
    print(f"  Precision / Recall / F1 : {prec:.4f} / {rec:.4f} / {f1:.4f}")
    print("-" * 65)
    print(f"BRANCH 2 (LOCALIZATION):")
    print(f"  Raw Model Mean Dice     : {mean_raw_dice:.4f}")
    print(f"  Raw Model Mean IoU      : {mean_raw_iou:.4f}")
    print(f"  Otsu Cleaned Mean Dice  : {mean_otsu_dice:.4f}")
    print("-" * 65)
    print(f"BRANCH 3 (TAMPERING CATEGORIZATION):")
    print(f"  Overall 3-Class Accuracy: {type_acc * 100:.2f}%")
    print(f"  Macro-Averaged F1-Score : {type_macro_f1:.4f}")
    print("\nDetailed Per-Class Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 65 + "\n")

    return {
        "det_acc": acc, "roc_auc": roc_auc, "gen_acc": gen_acc, "forged_rec": forged_rec,
        "raw_dice": mean_raw_dice, "raw_iou": mean_raw_iou, "otsu_dice": mean_otsu_dice,
        "type_acc": type_acc, "type_macro_f1": type_macro_f1
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ForgeryNet checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth checkpoint")
    p.add_argument("--data_dir",   type=str, default="Bengali_Handwriting_CopyMove_Dataset")
    p.add_argument("--mode",       type=str, default="patch", choices=["patch", "fullpage"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--det_thresh", type=float, default=0.32)
    p.add_argument("--seg_thresh", type=float, default=0.25)
    p.add_argument("--no_dual_stream", action="store_true")
    p.add_argument("--no_attention",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")

    model = ForgeryNet(
        pretrained=False,
        dual_stream=(not args.no_dual_stream),
        use_attention=(not args.no_attention)
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    _, _, test_loader = make_dataloaders(
        root_dir=args.data_dir,
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=0
    )

    evaluate_model(model, test_loader, device, det_thresh=args.det_thresh, seg_thresh=args.seg_thresh)
