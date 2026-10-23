"""
losses.py — Multi-task loss functions for ForgeryNet.

Supports both:
1. Baseline losses (unweighted BCE + unweighted Dice)
2. Advanced SOTA losses (Balanced BCE + Focal Tversky Loss with alpha=0.7, beta=0.3 + Masked CCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Focal Tversky Loss for extreme class imbalance (<1% tampered ink)
# -----------------------------------------------------------------------------

class FocalTverskyLoss(nn.Module):
    """
    Tversky Loss with Focal penalty to handle severe pixel sparsity.
    
    alpha=0.7 penalizes False Negatives (missed tampered strokes) over 2x more
    heavily than beta=0.3 False Positives (spurious activations on blank paper).
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred is sigmoid probability [0, 1]
        p = pred.contiguous().view(pred.size(0), -1)
        t = target.contiguous().view(target.size(0), -1)

        tp = (p * t).sum(dim=1)
        fp = (p * (1.0 - t)).sum(dim=1)
        fn = ((1.0 - p) * t).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        focal_tversky = torch.pow((1.0 - tversky), self.gamma)
        return focal_tversky.mean()


class UnweightedDiceLoss(nn.Module):
    """Standard unweighted soft Dice loss used in baseline experiments."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.contiguous().view(pred.size(0), -1)
        t = target.contiguous().view(target.size(0), -1)
        intersection = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


# -----------------------------------------------------------------------------
# Detection Loss (Branch 1)
# -----------------------------------------------------------------------------

def detection_loss(det_logit: torch.Tensor, binary_label: torch.Tensor, pos_weight: float = None) -> torch.Tensor:
    """
    Binary Cross-Entropy for Branch 1.
    If pos_weight is provided (e.g. 220/660 ~ 0.33), adjusts decision boundary.
    """
    target = binary_label.float().unsqueeze(1)
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=det_logit.device)
        return F.binary_cross_entropy_with_logits(det_logit, target, pos_weight=pw)
    return F.binary_cross_entropy_with_logits(det_logit, target)


# -----------------------------------------------------------------------------
# Composite Multi-Task Loss
# -----------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    Unified multi-task objective:
        L_total = lambda_det * L_det + lambda_loc * L_loc + lambda_type * L_type
    """
    def __init__(
        self,
        lambda_det: float = 1.0,
        lambda_loc: float = 4.0,
        lambda_type: float = 1.0,
        use_focal_tversky: bool = True,
        pos_weight_loc: float = 15.0,
        pos_weight_det: float = 0.33,
    ):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_loc = lambda_loc
        self.lambda_type = lambda_type
        self.use_focal_tversky = use_focal_tversky
        self.pos_weight_loc = pos_weight_loc
        self.pos_weight_det = pos_weight_det

        if use_focal_tversky:
            self.seg_criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
        else:
            self.seg_criterion = UnweightedDiceLoss()

    def forward(
        self,
        det_logit: torch.Tensor,
        seg_mask: torch.Tensor,
        type_logit: torch.Tensor,
        binary_label: torch.Tensor,
        type_label: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> dict:
        # 1. Branch 1: Binary Detection Loss
        l_det = detection_loss(det_logit, binary_label, pos_weight=self.pos_weight_det)

        # 2. Branch 2: Pixel-Level Localization Loss
        if self.use_focal_tversky:
            seg_logit = torch.logit(seg_mask.float().clamp(1e-6, 1.0 - 1e-6))
            pw = torch.tensor([self.pos_weight_loc], device=det_logit.device)
            wbce = F.binary_cross_entropy_with_logits(seg_logit, gt_mask.float(), pos_weight=pw)
            ftv  = self.seg_criterion(seg_mask.float(), gt_mask.float())
            l_loc = 0.5 * wbce + 0.5 * ftv
        else:
            seg_logit = torch.logit(seg_mask.float().clamp(1e-6, 1.0 - 1e-6))
            bce = F.binary_cross_entropy_with_logits(seg_logit, gt_mask.float())
            dice = self.seg_criterion(seg_mask.float(), gt_mask.float())
            l_loc = 0.5 * bce + 0.5 * dice

        # 3. Branch 3: Masked Tampering Categorization Loss (Forged samples only)
        forged_mask = (binary_label == 1).float()
        if forged_mask.sum() == 0:
            l_type = torch.tensor(0.0, device=type_logit.device, requires_grad=True)
        else:
            safe_labels = type_label.clone()
            safe_labels[safe_labels == -1] = 0
            per_sample = F.cross_entropy(type_logit, safe_labels, reduction="none")
            l_type = (per_sample * forged_mask).sum() / (forged_mask.sum() + 1e-8)

        total = self.lambda_det * l_det + self.lambda_loc * l_loc + self.lambda_type * l_type
        return {"total": total, "detect": l_det, "localize": l_loc, "type": l_type}
