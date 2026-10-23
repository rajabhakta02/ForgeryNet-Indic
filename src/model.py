"""
model.py — ForgeryNet: Dual-Stream Multi-task DenseNet-121 with Spatial Self-Attention.

Heads
-----
Branch 1 – Detection (binary):   GAP -> FC -> Sigmoid  -> (B, 1)
Branch 2 – Localization (seg):   U-Net decoder with skip connections -> (B, 1, H, W)
Branch 3 – Type classification:  Concat[GAP(s5_att), GAP(bridge)] -> FC -> Softmax -> (B, 3)

Ablation Switches
-----------------
dual_stream   : bool = True   (RGB + 3-channel Laplacian high-pass noise residuals)
use_attention : bool = True   (Scaled dot-product spatial self-attention at bottleneck)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet121_Weights


# -----------------------------------------------------------------------------
# High-Pass Residual Extractor (Laplacian Kernel)
# -----------------------------------------------------------------------------

class HighPassResidual(nn.Module):
    """
    Extracts high-frequency noise and edge anomalies using fixed forensic filters.
    Suppresses normal handwriting semantics and highlights tampered boundary discontinuities.
    """
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([
            [-1., -1., -1.],
            [-1.,  8., -1.],
            [-1., -1., -1.]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.conv2d(x, self.kernel, padding=1, groups=3)
        return torch.sigmoid(res)


# -----------------------------------------------------------------------------
# Spatial Self-Attention Bottleneck
# -----------------------------------------------------------------------------

class SpatialSelfAttention(nn.Module):
    """
    Numerically Stable Self-Attention module for global Copy-Move comparisons.
    Uses scaled dot-product (1/sqrt(d_k)) and explicit FP32 casting to prevent FP16 overflow.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 8
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        
        # Cast to float32 internally to prevent FP16 numerical overflow
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1).float()  # B x N x C'
        proj_key   = self.key_conv(x).view(B, -1, H * W).float()                    # B x C' x N
        
        # Scaled dot-product (1 / sqrt(d_k))
        scale = 1.0 / math.sqrt(self.inter_channels)
        energy = torch.bmm(proj_query, proj_key) * scale
        
        # Clamp to avoid extreme exponentials in softmax
        energy = torch.clamp(energy, min=-50.0, max=50.0)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(B, -1, H * W).float()                  # B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W).to(x.dtype)
        
        # Residual connection
        out = self.gamma * out + x
        return out


# -----------------------------------------------------------------------------
# Decoder Block (U-Net style)
# -----------------------------------------------------------------------------

class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
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
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -----------------------------------------------------------------------------
# DenseNet-121 Encoder Wrapper
# -----------------------------------------------------------------------------

class _DenseNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        dn = models.densenet121(weights=weights)
        f  = dn.features

        if in_channels == 6:
            # Dual-stream: 6 input channels (RGB + 3-channel Laplacian HPF)
            old_conv = f.conv0
            new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                new_conv.weight[:, 3:, :, :] = old_conv.weight
            self.layer0 = nn.Sequential(new_conv, f.norm0, f.relu0)
        else:
            self.layer0 = nn.Sequential(f.conv0, f.norm0, f.relu0)

        self.pool0  = f.pool0
        self.block1 = f.denseblock1
        self.trans1 = f.transition1
        self.block2 = f.denseblock2
        self.trans2 = f.transition2
        self.block3 = f.denseblock3
        self.trans3 = f.transition3
        self.block4 = f.denseblock4
        self.norm5  = f.norm5

    def forward(self, x: torch.Tensor):
        s1 = self.layer0(x)
        x  = self.pool0(s1)
        x  = self.block1(x); s2 = x; x = self.trans1(x)
        x  = self.block2(x); s3 = x; x = self.trans2(x)
        x  = self.block3(x); s4 = x; x = self.trans3(x)
        x  = self.block4(x)
        s5 = F.relu(self.norm5(x), inplace=True)
        return s1, s2, s3, s4, s5


# -----------------------------------------------------------------------------
# Main Multi-Task ForgeryNet Model
# -----------------------------------------------------------------------------

class ForgeryNet(nn.Module):
    """
    Unified multi-task architecture for document forgery forensics.

    Args:
        pretrained    : Load ImageNet pretrained weights for DenseNet-121
        num_types     : Number of tampering classes (default: 3)
        dual_stream   : Enable Laplacian high-pass residual front-end stream
        use_attention : Enable bottleneck spatial self-attention module
    """
    def __init__(
        self,
        pretrained: bool = True,
        num_types: int = 3,
        dual_stream: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.dual_stream = dual_stream
        self.use_attention = use_attention

        if self.dual_stream:
            self.hpf = HighPassResidual()
            self.encoder = _DenseNetEncoder(pretrained=pretrained, in_channels=6)
        else:
            self.hpf = None
            self.encoder = _DenseNetEncoder(pretrained=pretrained, in_channels=3)

        if self.use_attention:
            self.attention = SpatialSelfAttention(1024)
        else:
            self.attention = None

        # Decoder bridge and upsampling blocks
        self.bridge = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up4 = _UpBlock(in_ch=512,  skip_ch=1024, out_ch=512)
        self.up3 = _UpBlock(in_ch=512,  skip_ch=512,  out_ch=256)
        self.up2 = _UpBlock(in_ch=256,  skip_ch=256,  out_ch=128)
        self.up1 = _UpBlock(in_ch=128,  skip_ch=64,   out_ch=64)

        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Branch 1: Detection Head
        self.det_pool = nn.AdaptiveAvgPool2d(1)
        self.det_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        # Branch 3: Tampering Type Categorization Head
        self.type_pool = nn.AdaptiveAvgPool2d(1)
        self.type_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_types),
        )

    def forward(self, x: torch.Tensor):
        if self.dual_stream and self.hpf is not None:
            noise_res = self.hpf(x)
            x_in = torch.cat([x, noise_res], dim=1)
        else:
            x_in = x

        s1, s2, s3, s4, s5 = self.encoder(x_in)

        if self.use_attention and self.attention is not None:
            s5_att = self.attention(s5)
        else:
            s5_att = s5

        # Branch 1: Detection
        det_feat = self.det_pool(s5_att)
        det_logit = self.det_head(det_feat)

        # Bridge
        bridge = self.bridge(s5_att)

        # Branch 2: Stroke Localization
        d4 = self.up4(bridge, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        seg_mask = self.seg_head(d1)

        # Branch 3: Tampering Categorization
        t_enc = self.det_pool(s5_att).flatten(1)
        t_brg = self.type_pool(bridge).flatten(1)
        t_concat = torch.cat([t_enc, t_brg], dim=1)
        type_logit = self.type_head(t_concat)

        return det_logit, seg_mask, type_logit


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = ForgeryNet(pretrained=False, dual_stream=True, use_attention=True).to(device)
    dummy = torch.randn(2, 3, 512, 512, device=device)
    d, s, t = m(dummy)
    print("ForgeryNet SOTA Forward Pass Successful:")
    print("  Detection logit  :", d.shape)
    print("  Segmentation mask:", s.shape)
    print("  Type logits      :", t.shape)
