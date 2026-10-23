"""
evaluate.py — Comprehensive evaluation and inference for ForgeryNet.

Usage
-----
    # Full test-set evaluation with a trained checkpoint
    python src/evaluate.py --ckpt outputs/checkpoints/best.pth

    # Single-image inference
    python src/evaluate.py --ckpt outputs/checkpoints/best.pth --image path/to/img.tif

Metrics
-------
Branch 1 (Detection)  : Accuracy, Precision, Recall, F1, ROC-AUC
Branch 2 (Localization): Pixel IoU, Dice, Boundary F1
Branch 3 (Type)        : Per-class P/R/F1, Macro-F1, Confusion Matrix
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import make_dataloaders, ForgeryDataset, TARGET_SIZE, _image_transform
from src.model   import ForgeryNet
from src.utils   import visualize_prediction, seed_everything, TYPE_NAMES

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[evaluate] WARNING: scikit-learn not found; some metrics will be skipped.")


# -----------------------------------------------------------------------------
# Quick per-batch Dice (used inside train.py via import)
# -----------------------------------------------------------------------------

def _quick_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Per-sample soft Dice for a batch.  Returns shape (B,).
    Operates on CPU tensors.
    """
    pred = (pred_mask > 0.5).float().view(pred_mask.size(0), -1)
    gt   = gt_mask.float().view(gt_mask.size(0), -1)
    inter = (pred * gt).sum(dim=1)
    union = pred.sum(dim=1) + gt.sum(dim=1)
    return (2.0 * inter + 1.0) / (union + 1.0)


# -----------------------------------------------------------------------------
# Pixel-level segmentation metrics
# -----------------------------------------------------------------------------

def _seg_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    """
    Compute IoU, Dice, and a simple boundary F1 for one sample.

    Args:
        pred_bin : (H, W) binary array
        gt_bin   : (H, W) binary array

    Returns:
        dict with ``iou``, ``dice``, ``boundary_f1``
    """
    pred_bin = pred_bin.astype(bool)
    gt_bin   = gt_bin.astype(bool)

    tp = (pred_bin & gt_bin).sum()
    fp = (pred_bin & ~gt_bin).sum()
    fn = (~pred_bin & gt_bin).sum()

    iou  = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    # Simple boundary approximation via erosion diff
    def _boundary(mask: np.ndarray) -> np.ndarray:
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask, iterations=2)
        return mask & ~eroded

    try:
        pred_b = _boundary(pred_bin)
        gt_b   = _boundary(gt_bin)
        btp = (pred_b & gt_b).sum()
        bfp = (pred_b & ~gt_b).sum()
        bfn = (~pred_b & gt_b).sum()
        bprec = btp / (btp + bfp + 1e-8)
        brec  = btp / (btp + bfn + 1e-8)
        bF1   = 2 * bprec * brec / (bprec + brec + 1e-8)
    except Exception:
        bF1 = 0.0

    return {"iou": float(iou), "dice": float(dice), "boundary_f1": float(bF1)}


# -----------------------------------------------------------------------------
# Full test-set evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:    ForgeryNet,
    loader,
    device:   torch.device,
    save_dir: str | None = None,
    max_vis:  int = 10,
) -> dict:
    """
    Run inference over the entire loader and compute all metrics.

    Returns:
        dict of metric values for all three branches.
    """
    model.eval()

    # Accumulators
    det_probs,  det_gts  = [], []
    type_preds, type_gts = [], []
    seg_iou, seg_dice, seg_bF1 = [], [], []
    vis_count = 0

    for batch_idx, (images, bin_lbl, type_lbl, masks) in enumerate(loader):
        images   = images.to(device, non_blocking=True)
        bin_lbl  = bin_lbl.to(device, non_blocking=True)
        type_lbl = type_lbl.to(device, non_blocking=True)
        masks    = masks.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            det_logit, seg_mask, type_logit = model(images)

        # Detection
        det_prob = det_logit.sigmoid().squeeze(1).cpu().numpy()
        det_probs.append(det_prob)
        det_gts.append(bin_lbl.cpu().numpy())

        # Type (forged samples only)
        forged_idx = (bin_lbl == 1).cpu()
        if forged_idx.any():
            tp_pred = type_logit.argmax(1).cpu().numpy()[forged_idx.numpy()]
            tp_gt   = type_lbl.cpu().numpy()[forged_idx.numpy()]
            type_preds.append(tp_pred)
            type_gts.append(tp_gt)

        # Segmentation — per sample
        seg_np  = (seg_mask.cpu().squeeze(1).numpy() > 0.5).astype(np.uint8)
        mask_np = masks.cpu().squeeze(1).numpy().astype(np.uint8)
        for i in range(seg_np.shape[0]):
            m = _seg_metrics(seg_np[i], mask_np[i])
            seg_iou.append(m["iou"])
            seg_dice.append(m["dice"])
            seg_bF1.append(m["boundary_f1"])

        # Visualise a few samples
        if save_dir and vis_count < max_vis:
            for i in range(min(images.shape[0], max_vis - vis_count)):
                save_path = os.path.join(
                    save_dir, "predictions", f"vis_b{batch_idx}_s{i}.png"
                )
                visualize_prediction(
                    image      = images[i].cpu(),
                    gt_mask    = masks[i].cpu(),
                    pred_mask  = seg_mask[i].cpu(),
                    binary_gt  = int(bin_lbl[i].item()),
                    binary_pred= float(det_prob[i]),
                    type_gt    = int(type_lbl[i].item()) if type_lbl[i].item() >= 0 else 0,
                    type_pred  = int(type_logit[i].argmax().item()),
                    save_path  = save_path,
                )
                vis_count += 1

    # -- Aggregate -------------------------------------------------------------
    all_det_prob = np.concatenate(det_probs)
    all_det_gt   = np.concatenate(det_gts)
    all_det_pred = (all_det_prob > 0.5).astype(int)

    results = {}

    # Branch 1
    if HAS_SKLEARN:
        results["det_accuracy"]  = accuracy_score(all_det_gt, all_det_pred)
        results["det_precision"] = precision_score(all_det_gt, all_det_pred, zero_division=0)
        results["det_recall"]    = recall_score(all_det_gt, all_det_pred, zero_division=0)
        results["det_f1"]        = f1_score(all_det_gt, all_det_pred, zero_division=0)
        try:
            results["det_auc"]   = roc_auc_score(all_det_gt, all_det_prob)
        except Exception:
            results["det_auc"]   = float("nan")
    else:
        tp = ((all_det_pred == 1) & (all_det_gt == 1)).sum()
        fp = ((all_det_pred == 1) & (all_det_gt == 0)).sum()
        fn = ((all_det_pred == 0) & (all_det_gt == 1)).sum()
        tn = ((all_det_pred == 0) & (all_det_gt == 0)).sum()
        results["det_accuracy"]  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        results["det_precision"] = tp / (tp + fp + 1e-8)
        results["det_recall"]    = tp / (tp + fn + 1e-8)
        results["det_f1"]        = 2 * results["det_precision"] * results["det_recall"] / (
            results["det_precision"] + results["det_recall"] + 1e-8
        )
        results["det_auc"]       = float("nan")

    # Branch 2
    results["seg_mean_iou"]   = float(np.mean(seg_iou))
    results["seg_mean_dice"]  = float(np.mean(seg_dice))
    results["seg_boundary_f1"]= float(np.mean(seg_bF1))

    # Branch 3
    if type_preds:
        all_type_pred = np.concatenate(type_preds)
        all_type_gt   = np.concatenate(type_gts)

        if HAS_SKLEARN:
            results["type_macro_f1"] = f1_score(
                all_type_gt, all_type_pred, average="macro", zero_division=0
            )
            results["type_report"]   = classification_report(
                all_type_gt, all_type_pred,
                target_names=TYPE_NAMES, zero_division=0,
            )
            results["type_confusion_matrix"] = confusion_matrix(all_type_gt, all_type_pred)
        else:
            f1s = []
            for c in range(3):
                tp = ((all_type_pred == c) & (all_type_gt == c)).sum()
                fp = ((all_type_pred == c) & (all_type_gt != c)).sum()
                fn = ((all_type_pred != c) & (all_type_gt == c)).sum()
                prec = tp / (tp + fp + 1e-8)
                rec  = tp / (tp + fn + 1e-8)
                f1s.append(2 * prec * rec / (prec + rec + 1e-8))
            results["type_macro_f1"] = float(np.mean(f1s))
    else:
        results["type_macro_f1"] = float("nan")

    return results


# -----------------------------------------------------------------------------
# Single-image inference
# -----------------------------------------------------------------------------

@torch.no_grad()
def infer_single(
    model:     ForgeryNet,
    image_path: str,
    device:    torch.device,
    img_size:  tuple = TARGET_SIZE,
    threshold: float = 0.5,
) -> dict:
    """
    Run inference on one image file and return a result dict.

    Returns:
        {
          "is_forged": bool,
          "forgery_prob": float,
          "type": str,        (only meaningful if is_forged)
          "type_probs": list,
          "mask": np.ndarray  (H, W) binary
        }
    """
    img    = Image.open(image_path).convert("RGB")
    img    = img.resize((img_size[1], img_size[0]), Image.BILINEAR)
    tf     = _image_transform(augment=False)
    tensor = tf(img).unsqueeze(0).to(device)    # (1, 3, H, W)

    model.eval()
    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        det_logit, seg_mask, type_logit = model(tensor)

    forgery_prob = float(det_logit.sigmoid().item())
    is_forged    = forgery_prob > threshold
    type_probs   = torch.softmax(type_logit, dim=1).squeeze().cpu().numpy().tolist()
    type_pred    = int(np.argmax(type_probs))
    mask_np      = (seg_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    return {
        "is_forged":    is_forged,
        "forgery_prob": forgery_prob,
        "type":         TYPE_NAMES[type_pred] if is_forged else "N/A",
        "type_probs":   {n: round(p, 4) for n, p in zip(TYPE_NAMES, type_probs)},
        "mask":         mask_np,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate / Infer ForgeryNet")
    p.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to checkpoint .pth (e.g. outputs/checkpoints/best.pth)",
    )
    p.add_argument(
        "--data_dir", type=str,
        default=r"c:\Users\DEBJANI KAR\Desktop\Forgery Detection\Bengali_Handwriting_CopyMove_Dataset",
    )
    p.add_argument("--out_dir",   type=str, default="outputs")
    p.add_argument("--bs",        type=int, default=8)
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--img_size",  type=int, default=512)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument(
        "--image", type=str, default=None,
        help="Path to a single .tif image for inference mode",
    )
    p.add_argument("--save_vis",  action="store_true", help="Save prediction visualisations")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Device: {device}")

    # -- Load model ------------------------------------------------------------
    model = ForgeryNet(pretrained=False).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)    # handle both bare and wrapped checkpoints
    model.load_state_dict(state)
    print(f"[evaluate] Loaded checkpoint: {args.ckpt}")

    # -- Single-image inference mode -------------------------------------------
    if args.image:
        result = infer_single(model, args.image, device, img_size=(args.img_size, args.img_size))
        print("\n-- Inference Result ------------------------------")
        print(f"  Forged      : {result['is_forged']}  (prob={result['forgery_prob']:.4f})")
        print(f"  Forgery type: {result['type']}")
        print(f"  Type probs  : {result['type_probs']}")
        print(f"  Mask pixels : {result['mask'].sum()}")

        # Save mask as PNG
        mask_img  = Image.fromarray(result["mask"] * 255)
        mask_out  = os.path.join(args.out_dir, "single_mask.png")
        os.makedirs(args.out_dir, exist_ok=True)
        mask_img.save(mask_out)
        print(f"  Mask saved  -> {mask_out}")
        return

    # -- Full test-set evaluation ----------------------------------------------
    img_size = (args.img_size, args.img_size)
    _, _, test_loader = make_dataloaders(
        root_dir    = args.data_dir,
        batch_size  = args.bs,
        num_workers = args.workers,
        seed        = args.seed,
        img_size    = img_size,
    )

    save_dir = args.out_dir if args.save_vis else None
    results  = evaluate(model, test_loader, device, save_dir=save_dir)

    # -- Print report ----------------------------------------------------------
    print("\n-- Branch 1: Detection -------------------------------------------")
    print(f"  Accuracy  : {results['det_accuracy']:.4f}")
    print(f"  Precision : {results['det_precision']:.4f}")
    print(f"  Recall    : {results['det_recall']:.4f}")
    print(f"  F1        : {results['det_f1']:.4f}")
    print(f"  ROC-AUC   : {results['det_auc']:.4f}")

    print("\n-- Branch 2: Localization ----------------------------------------")
    print(f"  Mean IoU       : {results['seg_mean_iou']:.4f}")
    print(f"  Mean Dice      : {results['seg_mean_dice']:.4f}")
    print(f"  Boundary F1    : {results['seg_boundary_f1']:.4f}")

    print("\n-- Branch 3: Type Classification (forged samples only) -----------")
    print(f"  Macro F1  : {results['type_macro_f1']:.4f}")
    if "type_report" in results:
        print("\n" + results["type_report"])
    if "type_confusion_matrix" in results:
        print("  Confusion Matrix:")
        print(results["type_confusion_matrix"])


if __name__ == "__main__":
    main()
