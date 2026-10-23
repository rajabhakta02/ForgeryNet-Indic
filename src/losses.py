"""
losses.py — Combined multi-task loss functions.

    L_total = λ1 * L_detect + λ2 * L_localize + λ3 * L_type

L_detect   : Binary Cross-Entropy (on raw det_logit)
L_localize : 0.5 * DiceLoss + 0.5 * pixel-BCE  (handles extreme class imbalance)
L_type     : Categorical Cross-Entropy  — masked to 0 for genuine images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Dice Loss
# -----------------------------------------------------------------------------

def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice loss for segmentation.

    Args:
        pred   : (B, 1, H, W) — sigmoid-activated probability map [0, 1]
        target : (B, 1, H, W) — binary ground-truth mask {0, 1}
        smooth : Laplace smoothing to avoid zero-division

    Returns:
        Scalar loss (1 - mean-Dice).
    """
    pred   = pred.contiguous().view(pred.size(0), -1)      # (B, H*W)
    target = target.contiguous().view(target.size(0), -1)  # (B, H*W)

    intersection = (pred * target).sum(dim=1)
    union        = pred.sum(dim=1) + target.sum(dim=1)

    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_score.mean()


# -----------------------------------------------------------------------------
# Individual branch losses
# -----------------------------------------------------------------------------

def detection_loss(det_logit: torch.Tensor, binary_label: torch.Tensor) -> torch.Tensor:
    """
    Binary Cross-Entropy on raw logits (numerically stable).

    Args:
        det_logit    : (B, 1) — raw logit before sigmoid
        binary_label : (B,)   — {0, 1}

    Returns:
        Scalar BCE loss.
    """
    target = binary_label.float().unsqueeze(1)   # (B, 1)
    return F.binary_cross_entropy_with_logits(det_logit, target)


def localization_loss(
    seg_mask: torch.Tensor,
    gt_mask:  torch.Tensor,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combined Dice + BCE loss for pixel-level segmentation.

    Using Dice prevents the model from trivially predicting an all-black mask
    to achieve artificially high accuracy when forged pixels are sparse
    (e.g. a single added Bengali character).

    Args:
        seg_mask    : (B, 1, H, W) — sigmoid probability map [0, 1]
        gt_mask     : (B, 1, H, W) — binary ground-truth {0, 1}
        dice_weight : weight for Dice (pixel-BCE weight = 1 - dice_weight)

    Returns:
        Scalar localization loss.
    """
    bce  = F.binary_cross_entropy(seg_mask, gt_mask.float())
    dice = dice_loss(seg_mask, gt_mask.float())
    return dice_weight * dice + (1.0 - dice_weight) * bce


def type_loss(
    type_logit:   torch.Tensor,
    type_label:   torch.Tensor,
    binary_label: torch.Tensor,
) -> torch.Tensor:
    """
    Categorical Cross-Entropy for forgery-type classification, masked so
    genuine images (binary_label == 0) contribute zero gradient.

    Args:
        type_logit   : (B, 3) — raw logits (Copy-Move / Splicing / CharAdd)
        type_label   : (B,)   — {0, 1, 2}  (-1 for genuine, masked out)
        binary_label : (B,)   — {0, 1}     used to build the mask

    Returns:
        Scalar masked CCE loss.
    """
    forged_mask = (binary_label == 1).float()     # 1.0 where forged, 0.0 genuine

    if forged_mask.sum() == 0:
        # Edge case: entire batch is genuine — return 0 loss
        return torch.tensor(0.0, device=type_logit.device, requires_grad=True)

    # Replace -1 labels (genuine) with 0 so CrossEntropyLoss doesn't error.
    # The mask will zero out their contribution.
    safe_labels = type_label.clone()
    safe_labels[safe_labels == -1] = 0

    per_sample_loss = F.cross_entropy(type_logit, safe_labels, reduction="none")  # (B,)
    masked_loss     = per_sample_loss * forged_mask                                # zero genuine
    return masked_loss.sum() / (forged_mask.sum() + 1e-8)                          # mean over forged


# -----------------------------------------------------------------------------
# Combined loss
# -----------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    Weighted sum of the three branch losses.

    Default weights:
        λ1 = 1.0  (detection)
        λ2 = 2.0  (localization — higher weight to combat mask sparsity)
        λ3 = 1.0  (type classification)

    Args:
        lambda_det  : Weight for detection loss.
        lambda_loc  : Weight for localization loss.
        lambda_type : Weight for type-classification loss.
        dice_weight : Fraction of Dice in localization loss.
    """

    def __init__(
        self,
        lambda_det:  float = 1.0,
        lambda_loc:  float = 2.0,
        lambda_type: float = 1.0,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_det  = lambda_det
        self.lambda_loc  = lambda_loc
        self.lambda_type = lambda_type
        self.dice_weight = dice_weight

    def forward(
        self,
        det_logit:    torch.Tensor,   # (B, 1)
        seg_mask:     torch.Tensor,   # (B, 1, H, W)
        type_logit:   torch.Tensor,   # (B, 3)
        binary_label: torch.Tensor,   # (B,)
        type_label:   torch.Tensor,   # (B,)
        gt_mask:      torch.Tensor,   # (B, 1, H, W)
    ) -> dict:
        """
        Compute all partial losses and the weighted total.

        Returns a dict with keys:
            ``total``, ``detect``, ``localize``, ``type``
        """
        l_det  = detection_loss(det_logit, binary_label)
        l_loc  = localization_loss(seg_mask, gt_mask, self.dice_weight)
        l_type = type_loss(type_logit, type_label, binary_label)

        total = (
            self.lambda_det  * l_det
            + self.lambda_loc  * l_loc
            + self.lambda_type * l_type
        )

        return {
            "total":    total,
            "detect":   l_det,
            "localize": l_loc,
            "type":     l_type,
        }
