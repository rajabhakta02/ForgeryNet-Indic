"""
model.py — ForgeryNet: Multi-task DenseNet-121 backbone with three output heads.

Heads
-----
Branch 1 – Detection (binary):   GAP -> FC -> Sigmoid  -> (B, 1)
Branch 2 – Localization (seg):   U-Net decoder with skip connections -> (B, 1, H, W)
Branch 3 – Type classification:  Concat[bottleneck_pool, dec_bottleneck_pool]
                                  -> FC -> Softmax -> (B, 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet121_Weights


# -----------------------------------------------------------------------------
# Decoder block
# -----------------------------------------------------------------------------

class _UpBlock(nn.Module):
    """
    Transpose-conv upsample -> concatenate skip -> two Conv-BN-ReLU layers.

    Args:
        in_ch   : channels coming from the previous decoder level
        skip_ch : channels from the encoder skip connection
        out_ch  : output channels after this block
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_ch, in_ch // 2, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Align spatial dims (handles odd sizes)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -----------------------------------------------------------------------------
# DenseNet-121 feature extractor
# -----------------------------------------------------------------------------

class _DenseNetEncoder(nn.Module):
    """
    Wraps DenseNet-121's ``features`` sub-module and exposes 5 skip-connection
    feature maps at progressively halved spatial resolutions.

    Given input (B, 3, 512, 512):
        s1 : (B,  64,  128, 128)  – after initial conv+pool
        s2 : (B, 256,   64,  64)  – after dense-block-1 + transition-1
        s3 : (B, 512,   32,  32)  – after dense-block-2 + transition-2
        s4 : (B, 1024,  16,  16)  – after dense-block-3 + transition-3
        s5 : (B, 1024,   8,   8)  – after dense-block-4 (bottleneck)
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        dn = models.densenet121(weights=weights)
        f  = dn.features

        # Split the DenseNet feature sub-modules into named groups
        self.layer0 = nn.Sequential(f.conv0, f.norm0, f.relu0)   # 64ch, /2
        self.pool0  = f.pool0                                      # 64ch, /4

        self.block1 = f.denseblock1                                # 256ch
        self.trans1 = f.transition1                                # 128ch, /8

        self.block2 = f.denseblock2                                # 512ch
        self.trans2 = f.transition2                                # 256ch, /16

        self.block3 = f.denseblock3                                # 1024ch
        self.trans3 = f.transition3                                # 512ch, /32

        self.block4 = f.denseblock4                                # 1024ch
        self.norm5  = f.norm5

    def forward(self, x: torch.Tensor):
        # s1 : (B, 64,   H/4,  W/4)
        s1 = self.layer0(x)
        x  = self.pool0(s1)

        # s2 : (B, 256, H/8,  W/8)
        x  = self.block1(x)
        s2 = x
        x  = self.trans1(x)

        # s3 : (B, 512, H/16, W/16)
        x  = self.block2(x)
        s3 = x
        x  = self.trans2(x)

        # s4 : (B, 1024, H/32, W/32)
        x  = self.block3(x)
        s4 = x
        x  = self.trans3(x)

        # s5 / bottleneck : (B, 1024, H/64, W/64)
        x  = self.block4(x)
        s5 = F.relu(self.norm5(x), inplace=True)

        return s1, s2, s3, s4, s5


# -----------------------------------------------------------------------------
# Main multi-task model
# -----------------------------------------------------------------------------

class ForgeryNet(nn.Module):
    """
    Multi-task document-forgery detection network.

    Outputs (all as tensors, not necessarily sigmoid-activated):
        det_logit  : (B, 1)          — raw logit for BCE loss
        seg_mask   : (B, 1, H, W)    — sigmoid probability map [0, 1]
        type_logit : (B, 3)          — raw logits for CCE loss

    Notes
    -----
    * ``type_logit`` is always computed for the whole batch, but its
      contribution to the total loss is masked by ``binary_label`` in
      the training loop so genuine images receive zero type-loss gradient.
    * At inference, threshold ``det_logit.sigmoid() > 0.5`` to decide
      whether to trust ``type_logit``.
    """

    # Channel sizes at each skip connection in DenseNet-121
    _ENC_CHS = [64, 256, 512, 1024, 1024]   # s1..s5

    def __init__(self, pretrained: bool = True, num_types: int = 3) -> None:
        super().__init__()

        # -- Encoder ---------------------------------------------------------
        self.encoder = _DenseNetEncoder(pretrained=pretrained)

        # -- Bottleneck bridge (keeps s5 channels consistent) -----------------
        self.bridge = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # -- Decoder (U-Net style) --------------------------------------------
        # in_ch = bridge output (512), skip from s4 (1024), …
        self.up4 = _UpBlock(in_ch=512,  skip_ch=1024, out_ch=512)
        self.up3 = _UpBlock(in_ch=512,  skip_ch=512,  out_ch=256)
        self.up2 = _UpBlock(in_ch=256,  skip_ch=256,  out_ch=128)
        self.up1 = _UpBlock(in_ch=128,  skip_ch=64,   out_ch=64)

        # Final 1×1 conv + upsample to match input resolution
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # -- Branch 1 – Detection ---------------------------------------------
        self.det_pool = nn.AdaptiveAvgPool2d(1)
        self.det_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        # -- Branch 3 – Forgery type ------------------------------------------
        # Concatenates: GAP(s5=1024) + GAP(bridge_out=512) = 1536
        self.type_pool = nn.AdaptiveAvgPool2d(1)
        self.type_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_types),
        )

    def forward(self, x: torch.Tensor):
        # -- Encode -----------------------------------------------------------
        s1, s2, s3, s4, s5 = self.encoder(x)

        # -- Branch 1: Detection ----------------------------------------------
        det_feat  = self.det_pool(s5)            # (B, 1024, 1, 1)
        det_logit = self.det_head(det_feat)       # (B, 1)

        # -- Bottleneck bridge ------------------------------------------------
        bridge = self.bridge(s5)                  # (B, 512, H/64, W/64)

        # -- Branch 2: Localization decoder -----------------------------------
        d4 = self.up4(bridge, s4)                # (B, 512,  H/32, W/32)
        d3 = self.up3(d4, s3)                    # (B, 256,  H/16, W/16)
        d2 = self.up2(d3, s2)                    # (B, 128,  H/8,  W/8 )
        d1 = self.up1(d2, s1)                    # (B,  64,  H/4,  W/4 )
        seg_mask = self.seg_head(d1)             # (B,   1,  H,    W   )

        # -- Branch 3: Type classification ------------------------------------
        type_feat_enc    = self.det_pool(s5).flatten(1)     # (B, 1024)
        type_feat_bridge = self.type_pool(bridge).flatten(1)  # (B, 512)
        type_concat      = torch.cat([type_feat_enc, type_feat_bridge], dim=1)  # (B, 1536)
        type_logit       = self.type_head(type_concat)      # (B, 3)

        return det_logit, seg_mask, type_logit


# -----------------------------------------------------------------------------
# Quick sanity-check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ForgeryNet(pretrained=False).to(device)

    dummy  = torch.randn(2, 3, 512, 512, device=device)
    with torch.no_grad():
        d_logit, seg, t_logit = model(dummy)

    print("det_logit  :", d_logit.shape)   # (2, 1)
    print("seg_mask   :", seg.shape)        # (2, 1, 512, 512)
    print("type_logit :", t_logit.shape)   # (2, 3)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters : {total:.1f} M")
