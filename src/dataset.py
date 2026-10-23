"""
dataset.py — ForgeryDataset and DataLoader factory for the
Bengali Handwriting Forgery Detection pipeline.

Each sample yields:
    image       : FloatTensor (3, H, W)  — ImageNet-normalised
    binary_label: LongTensor  ()         — 0=Genuine, 1=Forged
    type_label  : LongTensor  ()         — 0=CopyMove, 1=Splicing, 2=CharAdd, -1=Genuine
    mask        : FloatTensor (1, H, W)  — binary {0,1}; all-zeros for genuine
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Forgery type label mapping
TYPE_LABEL = {
    "copy_move":          0,
    "splicing":           1,
    "character_addition": 2,
}

# Input spatial size fed to the network
TARGET_SIZE = (512, 512)   # (H, W)


# -----------------------------------------------------------------------------
# Sample collection helpers
# -----------------------------------------------------------------------------

def _collect_genuine(genuine_dir: str) -> List[Dict]:
    """Gather all genuine .tif files as label=0 samples."""
    samples = []
    for fname in sorted(os.listdir(genuine_dir)):
        if fname.lower().endswith(".tif"):
            samples.append({
                "image_path":   os.path.join(genuine_dir, fname),
                "mask_path":    None,           # will produce all-zero mask
                "binary_label": 0,
                "type_label":   -1,             # masked out during type-loss
            })
    return samples


def _collect_forged(category_dir: str, type_key: str) -> List[Dict]:
    """
    Gather forged samples from a category folder that follows the layout:
        category_dir/
            forged/    *.tif
            masks/     *_mask.tif
    Masks are matched by replacing the forged filename stem with '<stem>_mask'.
    """
    forged_dir = os.path.join(category_dir, "forged")
    mask_dir   = os.path.join(category_dir, "masks")
    type_lbl   = TYPE_LABEL[type_key]

    samples = []
    for fname in sorted(os.listdir(forged_dir)):
        if not fname.lower().endswith(".tif"):
            continue

        stem      = Path(fname).stem                    # e.g. AKD_copy_move_01_final
        mask_name = f"{stem}_mask.tif"
        mask_path = os.path.join(mask_dir, mask_name)

        # Gracefully handle a missing mask — treat as all-zero
        if not os.path.exists(mask_path):
            mask_path = None

        samples.append({
            "image_path":   os.path.join(forged_dir, fname),
            "mask_path":    mask_path,
            "binary_label": 1,
            "type_label":   type_lbl,
        })
    return samples


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------

def _image_transform(augment: bool) -> transforms.Compose:
    ops = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class ForgeryDataset(Dataset):
    """
    Unified dataset for Bengali Handwriting Forgery Detection.

    Args:
        root_dir  : Path to ``Bengali_Handwriting_CopyMove_Dataset/``
        samples   : Pre-split list of sample dicts (from ``make_dataloaders``).
        augment   : If True, applies training-time augmentation.
        img_size  : (H, W) to resize all images/masks to.
    """

    def __init__(
        self,
        root_dir:  str,
        samples:   List[Dict],
        augment:   bool = False,
        img_size:  Tuple[int, int] = TARGET_SIZE,
    ) -> None:
        self.samples   = samples
        self.augment   = augment
        self.img_size  = img_size
        self.img_tf    = _image_transform(augment)

    # -- Internals -------------------------------------------------------------

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        # Handle multi-page TIFs — take the first page
        try:
            img.seek(0)
        except Exception:
            pass
        return img.convert("RGB")

    def _load_mask(self, path: str | None) -> np.ndarray:
        """Return a (H, W) binary numpy array in {0, 1}."""
        if path is None or not os.path.exists(path):
            return np.zeros(self.img_size, dtype=np.float32)

        mask = Image.open(path).convert("L")            # grayscale
        mask = mask.resize(
            (self.img_size[1], self.img_size[0]),        # PIL takes (W, H)
            resample=Image.NEAREST,
        )
        arr = np.array(mask, dtype=np.float32)
        arr = (arr > 127).astype(np.float32)             # binarise
        return arr

    def _apply_joint_augment(
        self,
        image: Image.Image,
        mask_arr: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        """Apply the same spatial augmentations to image AND mask."""
        if not self.augment:
            return image, mask_arr

        # Horizontal flip
        if random.random() < 0.5:
            image    = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_arr = mask_arr[:, ::-1].copy()

        # Vertical flip
        if random.random() < 0.3:
            image    = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask_arr = mask_arr[::-1, :].copy()

        # Small rotation  (±5°)
        angle = random.uniform(-5, 5)
        image    = image.rotate(angle, resample=Image.BILINEAR, expand=False)
        mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8))
        mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST, expand=False)
        mask_arr = (np.array(mask_pil) > 127).astype(np.float32)

        return image, mask_arr

    # -- Public interface ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        s = self.samples[idx]

        # -- Load & resize image ----------------------------------------------
        image = self._load_image(s["image_path"])
        image = image.resize(
            (self.img_size[1], self.img_size[0]),   # PIL (W, H)
            resample=Image.BILINEAR,
        )

        # -- Load mask --------------------------------------------------------
        mask_arr = self._load_mask(s["mask_path"])

        # -- Joint spatial augmentation ---------------------------------------
        image, mask_arr = self._apply_joint_augment(image, mask_arr)

        # -- Convert to tensors -----------------------------------------------
        image_tensor = self.img_tf(image)                    # (3, H, W), float32
        mask_tensor  = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)

        binary_label = torch.tensor(s["binary_label"], dtype=torch.long)
        type_label   = torch.tensor(s["type_label"],   dtype=torch.long)

        return image_tensor, binary_label, type_label, mask_tensor


# -----------------------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------------------

def make_dataloaders(
    root_dir:   str,
    batch_size: int  = 8,
    num_workers: int = 4,
    seed:       int  = 42,
    img_size:   Tuple[int, int] = TARGET_SIZE,
    val_frac:   float = 0.15,
    test_frac:  float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build stratified train / val / test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset_dir = root_dir  # expects Bengali_Handwriting_CopyMove_Dataset/

    # -- Collect all samples ---------------------------------------------------
    all_samples: List[Dict] = []
    all_samples += _collect_genuine(os.path.join(dataset_dir, "genuine"))
    all_samples += _collect_forged(
        os.path.join(dataset_dir, "copy_move_clean"), "copy_move"
    )
    all_samples += _collect_forged(
        os.path.join(dataset_dir, "splicing"), "splicing"
    )
    all_samples += _collect_forged(
        os.path.join(dataset_dir, "character_addition"), "character_addition"
    )

    print(f"[dataset] Total samples collected: {len(all_samples)}")

    # -- Stratified split by binary label -------------------------------------
    genuine_s = [s for s in all_samples if s["binary_label"] == 0]
    forged_s  = [s for s in all_samples if s["binary_label"] == 1]

    def split_list(lst: List, val_f: float, test_f: float):
        random.shuffle(lst)
        n = len(lst)
        n_test = max(1, int(n * test_f))
        n_val  = max(1, int(n * val_f))
        return lst[n_test + n_val:], lst[n_test:n_test + n_val], lst[:n_test]

    g_train, g_val, g_test = split_list(genuine_s, val_frac, test_frac)
    f_train, f_val, f_test = split_list(forged_s,  val_frac, test_frac)

    train_samples = g_train + f_train
    val_samples   = g_val   + f_val
    test_samples  = g_test  + f_test

    random.shuffle(train_samples)

    print(f"[dataset] Split -> train:{len(train_samples)}  "
          f"val:{len(val_samples)}  test:{len(test_samples)}")

    # -- Instantiate Dataset objects -------------------------------------------
    train_ds = ForgeryDataset(dataset_dir, train_samples, augment=True,  img_size=img_size)
    val_ds   = ForgeryDataset(dataset_dir, val_samples,   augment=False, img_size=img_size)
    test_ds  = ForgeryDataset(dataset_dir, test_samples,  augment=False, img_size=img_size)

    loader_kw = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader
