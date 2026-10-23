"""
experiments/stage3_dualstream_attention.py — Reproduces Row 3 of Table 1 in the paper.

Configuration:
- Full-page downsampling to 512x512
- Dual-Stream Front-End (RGB + 3-channel Laplacian high-pass residuals)
- Bottleneck Non-local Spatial Self-Attention (scaled dot-product with clamping)
- Focal Tversky Loss (alpha=0.7, beta=0.3)
- Surges Branch 1 ROC-AUC from 0.7853 to 0.8606
"""

import subprocess
import sys

if __name__ == "__main__":
    cmd = [
        sys.executable, "src/train.py",
        "--stage", "3",
        "--out_dir", "outputs/stage3_dualstream_attention",
        "--epochs", "35",
        "--bs", "8"
    ]
    print("Launching Stage 3 (Dual-Stream + Attention) Experiment...")
    sys.exit(subprocess.call(cmd))
