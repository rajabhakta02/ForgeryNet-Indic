"""
experiments/stage2_balanced_loss.py — Reproduces Row 2 of Table 1 in the paper.

Configuration:
- Full-page downsampling to 512x512
- Standard DenseNet-121 RGB backbone
- Focal Tversky Loss (alpha=0.7, beta=0.3) + Positive-weighted BCE (w_loc=15.0, w_det=0.33)
- Resolves Branch 2 localization collapse (Mean Dice climbs from 0.000 to 0.3184)
"""

import subprocess
import sys

if __name__ == "__main__":
    cmd = [
        sys.executable, "src/train.py",
        "--stage", "2",
        "--out_dir", "outputs/stage2_balanced_loss",
        "--epochs", "35",
        "--bs", "8"
    ]
    print("Launching Stage 2 (Balanced Loss) Experiment...")
    sys.exit(subprocess.call(cmd))
