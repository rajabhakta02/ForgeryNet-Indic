"""
experiments/stage4_native_patch_sota.py — Reproduces Row 4 (Final Proposed SOTA) of Table 1 in the paper.

Configuration:
- Native-Resolution (512x512) patch crops extracted directly at 100% scan fidelity
- 70% biased sampling centered around active manipulation ink boundaries
- Dual-Stream Front-End (RGB + Laplacian noise residuals)
- Numerically Stable Spatial Self-Attention at bottleneck
- Multi-Task Loss: Focal Tversky (alpha=0.7, beta=0.3) + Balanced BCE + Masked CCE
- Benchmark Results:
    * Branch 1: 89.39% Detection Accuracy, 0.9446 ROC-AUC, 100.0% Genuine Specificity
    * Branch 2: 0.3330 Raw Mean Dice
    * Branch 3: 82.47% Categorization Accuracy, 0.8005 Macro F1, 1.00 Char-Addition F1
"""

import subprocess
import sys

if __name__ == "__main__":
    cmd = [
        sys.executable, "src/train.py",
        "--stage", "4",
        "--out_dir", "outputs/stage4_native_patch_sota",
        "--epochs", "35",
        "--bs", "8",
        "--lr", "1.5e-4"
    ]
    print("Launching Stage 4 (Final SOTA Native Patch Pipeline) Experiment...")
    sys.exit(subprocess.call(cmd))
