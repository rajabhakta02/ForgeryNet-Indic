"""
utils.py — Helper utilities for seed setting, visualization, and metric tracking.
"""

import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

TYPE_NAMES = ["Copy-Move", "Splicing", "Character Addition"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW image tensor back to a displayable HWC uint8 array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)   # CHW -> HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def visualize_prediction(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    binary_gt: int,
    binary_pred: float,
    type_gt: int,
    type_pred: int,
    save_path: str | None = None,
) -> None:
    """
    Plot a 4-panel figure:
      Col 1 – Original image
      Col 2 – Ground-truth mask (green overlay)
      Col 3 – Predicted mask (red overlay)
      Col 4 – Side-by-side overlay diff
    """
    img_np   = denormalize(image)
    gt_np    = gt_mask.squeeze().cpu().numpy()
    pred_np  = pred_mask.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # -- Original image ------------------------------------------------------
    axes[0].imshow(img_np)
    gt_str   = "Genuine" if binary_gt == 0 else TYPE_NAMES[type_gt]
    pred_str = "Genuine" if binary_pred < 0.5 else TYPE_NAMES[type_pred]
    axes[0].set_title(f"Input\nGT: {gt_str} | Pred: {pred_str}", fontsize=9)
    axes[0].axis("off")

    # -- Ground-truth mask overlay --------------------------------------------
    axes[1].imshow(img_np)
    gt_overlay = np.zeros((*gt_np.shape, 4), dtype=np.float32)
    gt_overlay[gt_np > 0.5] = [0, 1, 0, 0.4]   # green
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground-Truth Mask", fontsize=9)
    axes[1].axis("off")

    # -- Predicted mask overlay -----------------------------------------------
    axes[2].imshow(img_np)
    pred_overlay = np.zeros((*pred_np.shape, 4), dtype=np.float32)
    pred_overlay[pred_np > 0.5] = [1, 0, 0, 0.4]  # red
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Predicted Mask", fontsize=9)
    axes[2].axis("off")

    # -- Diff overlay ---------------------------------------------------------
    axes[3].imshow(img_np)
    diff_overlay = np.zeros((*gt_np.shape, 4), dtype=np.float32)
    tp = (gt_np > 0.5) & (pred_np > 0.5)
    fp = (gt_np <= 0.5) & (pred_np > 0.5)
    fn = (gt_np > 0.5) & (pred_np <= 0.5)
    diff_overlay[tp] = [0, 1, 0, 0.5]   # TP -> green
    diff_overlay[fp] = [1, 0, 0, 0.5]   # FP -> red
    diff_overlay[fn] = [0, 0, 1, 0.5]   # FN -> blue
    axes[3].imshow(diff_overlay)
    patches = [
        mpatches.Patch(color="green", label="TP"),
        mpatches.Patch(color="red",   label="FP"),
        mpatches.Patch(color="blue",  label="FN"),
    ]
    axes[3].legend(handles=patches, loc="upper right", fontsize=7)
    axes[3].set_title("TP/FP/FN Diff", fontsize=9)
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_training_curves(log_csv: str, save_dir: str = "outputs") -> None:
    """
    Parse the per-epoch CSV written by the training loop and plot loss /
    metric curves.  Expects columns: epoch, train_loss, val_loss,
    val_det_f1, val_dice, val_type_f1.
    """
    import pandas as pd

    df = pd.read_csv(log_csv)
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Detection F1
    axes[1].plot(df["epoch"], df["val_det_f1"], label="Val Det-F1", color="green")
    axes[1].set_title("Detection F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Localization Dice + Type F1
    axes[2].plot(df["epoch"], df["val_dice"],     label="Val Dice",    color="orange")
    axes[2].plot(df["epoch"], df["val_type_f1"],  label="Val Type-F1", color="purple")
    axes[2].set_title("Localization Dice & Type-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"[utils] Training curves saved -> {save_dir}/training_curves.png")
