"""
experiments/stage1_baseline.py — Reproduces Row 1 of Table 1 in the paper.

Configuration:
- Full-page downsampling to 512x512
- Standard DenseNet-121 RGB backbone (no noise residual stream)
- No spatial self-attention
- Unweighted BCE loss + Unweighted Dice loss
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    cmd = [
        sys.executable, "src/train.py",
        "--stage", "1",
        "--out_dir", "outputs/stage1_baseline",
        "--epochs", "35",
        "--bs", "8"
    ]
    print("Launching Stage 1 (Baseline CNN) Experiment...")
    sys.exit(subprocess.call(cmd))
