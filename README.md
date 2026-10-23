# ForgeryNet: Dual-Stream Multi-Task Indic Handwriting Forgery Forensics

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: IJDAR](https://img.shields.io/badge/Paper-IJDAR%20Submission-blue.svg)](manuscript.tex)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](ForgeryNet_Colab.ipynb)

Official code and experimental replication suite for **ForgeryNet**, a unified multi-task deep learning framework engineered for simultaneous **detection**, **pixel-level stroke localization**, and **tampering categorization** of handwritten documents in Indic scripts (evaluated on Bengali).

---

## 📌 Key Architectural Highlights

* **Dual-Stream Forensic Extractor**: Combines native RGB representations with a depthwise Laplacian high-frequency edge-discontinuity residual stream ($\mathbf{K}_{\text{HPF}}$).
* **DenseNet-121 Encoder Hierarchy**: Retains fine-grained, high-frequency stroke gradients across downsampling stages.
* **Spatial Self-Attention Bottleneck**: Scaled dot-product self-attention enabling non-local feature matching for long-range duplicated text (Copy-Move).
* **4-Stage U-Net Localization Decoder**: Recovers precise stroke boundaries using skip connections and transposed convolutions.
* **Focal Tversky Loss ($\alpha=0.7, \beta=0.3$)**: Overcomes extreme foreground-background class imbalance ($<1\%$ tampered ink), eliminating trivial zero-mask collapse.
* **Native-Resolution Patch Pipeline**: Operates on $512 \times 512$ crops at $100\%$ scan fidelity to preserve microscopic ink feathering, pen pressure, and stroke striations.

```
Input: Native Document Crop (3, 512, 512)
  │
  ├── High-Pass Filter (Laplacian 3x3) ──> Noise Residuals (3, 512, 512)
  │                                                │
  └────────────────── Concat ──────────────────────┘
                         │
                    (6, 512, 512)
                         ▼
        DenseNet-121 Feature Extractor
          s1: (  64, 128, 128)
          s2: ( 256,  64,  64)
          s3: ( 512,  32,  32)
          s4: (1024,  16,  16)
          s5: (1024,   8,   8)  <-- Bottleneck
                         │
                         ▼
          Spatial Self-Attention (Scaled Dot-Product)
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
    Bridge (Conv2D)                GAP + Flatten
         │                               │
    U-Net Decoder                        ├───────────┐
  (Up4 -> Up3 -> Up2 -> Up1)             ▼           ▼
         │                          Branch 1    Branch 3
         ▼                         (Detection)   (Type)
     Branch 2                        (B, 1)       (B, 3)
   (Localization)
   (1, 512, 512)
```

---

## 📊 Benchmark Results & Ablation Progression (Table 1 & Table 2)

Evaluated on the held-out test split of 132 Bengali handwriting documents (35 Genuine, 97 Forged):

| Experimental Configuration | Det. Acc (%) | ROC-AUC | Genuine Specificity | Forged Recall | Raw Mean Dice | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Baseline CNN** ($512^2$ Downsampled) | 75.00% | 0.7815 | 0.0% (0/33) | **100.0%** (99/99) | 0.0000 (Collapsed) | 0.7789 |
| **Stage 2: Balanced Loss** (Focal Tversky + BCE) | 72.00% | 0.7853 | 69.7% (23/33) | 72.7% (72/99) | 0.3184 | 0.7993 |
| **Stage 3: Dual-Stream + Attention** | 74.24% | 0.8606 | 18.2% (6/33) | 92.9% (92/99) | 0.0988 | 0.7666 |
| **Stage 4: Native Patch Pipeline (Final Proposed SOTA)** | **`89.39%`** | **`0.9446`** | **`100.0%` (35/35, 0 false alarms)** | 85.6% (83/97) | **`0.3330`** | **`0.8005`** |

### Branch 3: Fine-Grained Tampering Classification Report ($N = 97$)
* **Character Addition**: **`1.00 Precision, 1.00 Recall, 1.00 F1-Score`** (36/36 perfect classifications)
* **Splicing**: **`0.70 Precision, 0.88 Recall, 0.78 F1-Score`** (30/34 intercepted)
* **Copy-Move**: **`0.78 Precision, 0.52 Recall, 0.62 F1-Score`**
* **Overall 3-Class Accuracy**: **`82.47%`**

---

## 📂 Repository Organization

```
ForgeryNet/
├── src/
│   ├── model.py                # Dual-stream architecture with Laplacian filter & spatial self-attention
│   ├── dataset.py              # FullPage and NativePatch datasets with biased sampling
│   ├── losses.py               # Focal Tversky Loss, Balanced BCE, and Masked Categorical Cross-Entropy
│   ├── train.py                # Unified CLI trainer supporting Stages 1, 2, 3, and 4
│   ├── evaluate.py             # Full multi-task evaluation suite (ROC-AUC, Dice, Confusion Matrix)
│   ├── inference_full_page.py  # Full-page visual forensic inspection and bounding box overlays
│   └── utils.py                # Helper utilities, threshold tuning, and plotting
├── experiments/                # Dedicated scripts replicating Table 1 row-by-row
│   ├── stage1_baseline.py      # Replicates Row 1 (Baseline CNN downsampled)
│   ├── stage2_balanced_loss.py # Replicates Row 2 (Focal Tversky balanced loss)
│   ├── stage3_dualstream_attention.py # Replicates Row 3 (Dual-Stream + Attention)
│   └── stage4_native_patch_sota.py    # Replicates Row 4 (Final Proposed SOTA)
├── manuscript.tex              # Complete LaTeX manuscript formatted for Springer Nature / IJDAR
├── sn-bibliography.bib         # BibTeX literature references
├── sn-jnl.cls                  # Official Springer Nature document class
├── sn-mathphys-num.bst         # Springer Nature bibliography style
├── ForgeryNet_Colab.ipynb      # Interactive Google Colab reproduction notebook
├── DOCUMENTATION.md            # Detailed engineering log, training trajectories, and diagnostics
└── requirements.txt            # Python dependencies
```

---

## 🔬 Reproducing Training & Experiments

### 1. Requirements & Setup
```bash
git clone https://github.com/rajabhakta02/ForgeryNet-Indic.git
cd ForgeryNet-Indic
pip install -r requirements.txt
```

### 2. Replicate Individual Ablation Stages (Table 1)
To train and verify any specific ablation milestone:
```bash
# Stage 1: Baseline CNN
python experiments/stage1_baseline.py

# Stage 2: Balanced Loss
python experiments/stage2_balanced_loss.py

# Stage 3: Dual-Stream Forensics + Spatial Self-Attention
python experiments/stage3_dualstream_attention.py

# Stage 4: Final Proposed SOTA (Native-Resolution Patches)
python experiments/stage4_native_patch_sota.py
```

Or using the unified trainer directly:
```bash
python src/train.py --stage 4 --data_dir /path/to/Bengali_Handwriting_CopyMove_Dataset --epochs 35 --bs 8
```

### 3. Evaluate a Trained Checkpoint
```bash
python src/evaluate.py \
  --checkpoint outputs/stage_4/checkpoints/best.pth \
  --data_dir /path/to/Bengali_Handwriting_CopyMove_Dataset \
  --mode patch \
  --det_thresh 0.32 \
  --seg_thresh 0.25
```

### 4. Run Full-Document Visual Inspection
To produce a 4-panel forensic diagnostic image with bounding boxes:
```bash
python src/inference_full_page.py \
  --image /path/to/document.tif \
  --mask /path/to/ground_truth_mask.tif \
  --checkpoint outputs/stage_4/checkpoints/best.pth \
  --out forensic_inspection_report.png
```

---

## 📄 Citation
```bibtex
@article{forgerynet2026,
  author    = {Author, First and Author, Second},
  title     = {ForgeryNet: A Dual-Stream Multi-Task Deep Framework for Detection, Localization, and Categorization of Handwritten Forgeries in Indic Scripts},
  journal   = {International Journal on Document Analysis and Recognition (IJDAR)},
  year      = {2026}
}
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
