"""
dataset.py — Dataset loaders for Bengali Handwriting Forgery Detection.

Provides:
1. FullPageForgeryDataset: Resizes full document scans to (512, 512).
2. PatchForgeryDataset: Extracts native-resolution (512, 512) crops with 70%
   biased sampling centered on active tampering stroke coordinates.
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TYPE_LABEL    = {"copy_move": 0, "splicing": 1, "character_addition": 2}
TARGET_SIZE   = 512


# -----------------------------------------------------------------------------
# File Discovery & Stratified Split
# -----------------------------------------------------------------------------

def collect_samples(root_dir: str) -> List[Dict]:
    samples = []
    root = Path(root_dir)

    # 1. Genuine (binary=0, type=-1)
    gen_dir = root / "genuine"
    if gen_dir.exists():
        for f in sorted(os.listdir(gen_dir)):
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg")):
                samples.append({
                    "image_path": str(gen_dir / f),
                    "mask_path": None,
                    "binary_label": 0,
                    "type_label": -1,
                    "category": "genuine"
                })

    # 2. Forged categories (binary=1, type=0, 1, 2)
    for cat_name, type_idx in TYPE_LABEL.items():
        f_dir = root / cat_name / "forged"
        m_dir = root / cat_name / "masks"
        if not f_dir.exists():
            continue
        for f in sorted(os.listdir(f_dir)):
            if not f.lower().endswith((".tif", ".tiff", ".png", ".jpg")):
                continue
            stem = Path(f).stem
            mask_path = m_dir / f"{stem}_mask.tif"
            if not mask_path.exists():
                mask_path = m_dir / f"{stem}_mask.png"
            samples.append({
                "image_path": str(f_dir / f),
                "mask_path": str(mask_path) if mask_path.exists() else None,
                "binary_label": 1,
                "type_label": type_idx,
                "category": cat_name
            })
    return samples


def split_samples(
    samples: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Performs category-stratified splitting across train (70%), val (15%), test (15%)."""
    rng = random.Random(seed)
    by_cat = {}
    for s in samples:
        by_cat.setdefault(s["category"], []).append(s)

    train, val, test = [], [], []
    for cat, items in by_cat.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_tr = int(n * train_ratio)
        n_vl = int(n * val_ratio)
        train.extend(shuffled[:n_tr])
        val.extend(shuffled[n_tr:n_tr + n_vl])
        test.extend(shuffled[n_tr + n_vl:])

    return train, val, test


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------

def get_transforms(augment: bool = False):
    ops = []
    if augment:
        ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return transforms.Compose(ops)


# -----------------------------------------------------------------------------
# Dataset 1: Full-Page Resized Dataset (Stages 1, 2, 3)
# -----------------------------------------------------------------------------

class FullPageForgeryDataset(Dataset):
    def __init__(self, samples: List[Dict], img_size: int = TARGET_SIZE, augment: bool = False):
        self.samples = samples
        self.img_size = img_size
        self.augment = augment
        self.tf = get_transforms(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        try: img.seek(0)
        except Exception: pass

        if item["mask_path"] and os.path.exists(item["mask_path"]):
            mask = Image.open(item["mask_path"]).convert("L")
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask_arr = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        else:
            mask_arr = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask_arr = np.fliplr(mask_arr).copy()

        img_t = self.tf(img)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)
        return img_t, torch.tensor(item["binary_label"], dtype=torch.long), torch.tensor(item["type_label"], dtype=torch.long), mask_t


# -----------------------------------------------------------------------------
# Dataset 2: Native-Resolution Biased Patch Dataset (Stage 4 - Final SOTA)
# -----------------------------------------------------------------------------

class PatchForgeryDataset(Dataset):
    def __init__(self, samples: List[Dict], patch_size: int = TARGET_SIZE, augment: bool = False):
        self.samples = samples
        self.patch_size = patch_size
        self.augment = augment
        self.tf = get_transforms(augment)

    def __len__(self):
        return len(self.samples)

    def _get_crop_coords(self, W: int, H: int, mask_arr: np.ndarray) -> Tuple[int, int, int, int]:
        ps = self.patch_size
        if W <= ps or H <= ps:
            return 0, 0, min(W, ps), min(H, ps)

        has_forgery = mask_arr.sum() > 0
        if has_forgery and (random.random() < 0.70 or not self.augment):
            y_indices, x_indices = np.where(mask_arr > 0)
            c_idx = random.randint(0, len(y_indices) - 1)
            cx, cy = int(x_indices[c_idx]), int(y_indices[c_idx])
            x0 = max(0, min(W - ps, cx - ps // 2 + random.randint(-40, 40)))
            y0 = max(0, min(H - ps, cy - ps // 2 + random.randint(-40, 40)))
        else:
            x0 = random.randint(0, W - ps)
            y0 = random.randint(0, H - ps)
        return x0, y0, x0 + ps, y0 + ps

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        try: img.seek(0)
        except Exception: pass
        W, H = img.size

        if item["mask_path"] and os.path.exists(item["mask_path"]):
            mask = Image.open(item["mask_path"]).convert("L")
            mask_arr = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        else:
            mask_arr = np.zeros((H, W), dtype=np.float32)

        x0, y0, x1, y1 = self._get_crop_coords(W, H, mask_arr)
        crop_img = img.crop((x0, y0, x1, y1))
        crop_mask = mask_arr[y0:y1, x0:x1]

        if crop_img.size != (self.patch_size, self.patch_size):
            crop_img = crop_img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            crop_mask = Image.fromarray((crop_mask * 255).astype(np.uint8)).resize((self.patch_size, self.patch_size), Image.NEAREST)
            crop_mask = (np.array(crop_mask, dtype=np.float32) > 127).astype(np.float32)

        if self.augment and random.random() > 0.5:
            crop_img = crop_img.transpose(Image.FLIP_LEFT_RIGHT)
            crop_mask = np.fliplr(crop_mask).copy()

        img_t = self.tf(crop_img)
        mask_t = torch.from_numpy(crop_mask).unsqueeze(0)
        return img_t, torch.tensor(item["binary_label"], dtype=torch.long), torch.tensor(item["type_label"], dtype=torch.long), mask_t


# -----------------------------------------------------------------------------
# Factory Loader
# -----------------------------------------------------------------------------

def make_dataloaders(
    root_dir: str,
    mode: str = "patch",
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Args:
        mode: 'patch' for native-resolution crops (Stage 4) or 'fullpage' for resized scans (Stages 1-3).
    """
    samples = collect_samples(root_dir)
    train_s, val_s, test_s = split_samples(samples, seed=seed)

    if mode == "patch":
        tr_ds = PatchForgeryDataset(train_s, augment=True)
        vl_ds = PatchForgeryDataset(val_s, augment=False)
        ts_ds = PatchForgeryDataset(test_s, augment=False)
    else:
        tr_ds = FullPageForgeryDataset(train_s, augment=True)
        vl_ds = FullPageForgeryDataset(val_s, augment=False)
        ts_ds = FullPageForgeryDataset(test_s, augment=False)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    ts_loader = DataLoader(ts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return tr_loader, vl_loader, ts_loader
