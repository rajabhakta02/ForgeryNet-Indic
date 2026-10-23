"""
inference_full_page.py — Generates full-page visual forensic inspection reports.

Takes a high-resolution full-page document, slides across patches,
computes anomaly logits, localizes manipulated strokes, and renders:
1. Original scan
2. Ground-truth mask
3. Predicted forgery probability mask
4. Bounding box & stroke overlay marking exact manipulated regions
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from torchvision import transforms
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ForgeryNet
from src.utils import TYPE_NAMES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def inspect_document(
    model: ForgeryNet,
    image_path: str,
    mask_path: str = None,
    out_path: str = "forensic_report.png",
    device: torch.device = torch.device("cpu"),
    patch_size: int = 512,
    det_thresh: float = 0.32,
    seg_thresh: float = 0.25,
):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    try: img.seek(0)
    except Exception: pass
    W, H = img.size

    if mask_path and os.path.exists(mask_path):
        gt_mask = Image.open(mask_path).convert("L")
        gt_arr = (np.array(gt_mask, dtype=np.float32) > 127).astype(np.float32)
    else:
        gt_arr = np.zeros((H, W), dtype=np.float32)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    pred_mask_full = np.zeros((H, W), dtype=np.float32)
    weight_full    = np.zeros((H, W), dtype=np.float32)
    step = patch_size // 2

    max_det_prob = 0.0
    detected_types = []

    # Sliding window inference
    with torch.no_grad():
        for y in range(0, max(1, H - patch_size + 1), step):
            for x in range(0, max(1, W - patch_size + 1), step):
                crop = img.crop((x, y, x + patch_size, y + patch_size))
                crop_t = tf(crop).unsqueeze(0).to(device)

                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    d_logit, s_mask, t_logit = model(crop_t)

                prob = torch.sigmoid(d_logit).item()
                if prob > max_det_prob:
                    max_det_prob = prob

                if prob > det_thresh:
                    t_idx = torch.argmax(t_logit, dim=1).item()
                    detected_types.append(t_idx)
                    s_np = s_mask[0, 0].cpu().numpy().astype(np.float32)
                    pred_mask_full[y:y + patch_size, x:x + patch_size] += s_np
                    weight_full[y:y + patch_size, x:x + patch_size] += 1.0

    weight_full[weight_full == 0] = 1.0
    pred_mask_full /= weight_full

    is_forged = max_det_prob > det_thresh
    predicted_cat = TYPE_NAMES[max(set(detected_types), key=detected_types.count)] if detected_types else "Authentic"

    # Render forensic 4-panel report
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img)
    axes[0].set_title(f"Input Document ({W}x{H})\nVerdict: {'FORGED' if is_forged else 'GENUINE'} ({max_det_prob:.2f})")
    axes[0].axis("off")

    axes[1].imshow(gt_arr, cmap="gray")
    axes[1].set_title("Ground-Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask_full, cmap="hot")
    axes[2].set_title(f"Predicted Probability Map\nType: {predicted_cat}")
    axes[2].axis("off")

    # Overlay with bounding boxes
    axes[3].imshow(img)
    axes[3].set_title("Forensic Inspection Overlay")
    if is_forged:
        bin_mask = (pred_mask_full > seg_thresh).astype(np.uint8)
        lbl = label(bin_mask)
        for prop in regionprops(lbl):
            if prop.area >= 50:
                minr, minc, maxr, maxc = prop.bbox
                rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                         fill=False, edgecolor="red", linewidth=2)
                axes[3].add_patch(rect)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Report saved to: {out_path} (Verdict: {'FORGED' if is_forged else 'GENUINE'}, Category: {predicted_cat})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to document scan")
    p.add_argument("--mask", default=None, help="Path to ground truth mask")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pth")
    p.add_argument("--out", default="forensic_report.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForgeryNet(pretrained=False, dual_stream=True, use_attention=True).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    inspect_document(model, args.image, args.mask, out_path=args.out, device=device)
