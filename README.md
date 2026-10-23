# ForgeryNet: Dual-Stream Multi-Task Indic Handwriting Forgery Forensics

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: IJDAR](https://img.shields.io/badge/Paper-IJDAR%20Submission-blue.svg)](manuscript.tex)

Official implementation of **ForgeryNet**, a unified multi-task deep learning framework engineered for simultaneous **detection**, **pixel-level stroke localization**, and **tampering categorization** of handwritten documents in Indic scripts (evaluated on Bengali).

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

## 📊 Benchmark Results

Evaluated on the held-out test split of the Bengali Handwriting Copy-Move & Tampering Dataset:

| Metric | Baseline CNN | Balanced Loss | Dual-Stream + Attn | **Native-Resolution ForgeryNet (Final)** |
| :--- | :---: | :---: | :---: | :---: |
| **Branch 1: Overall Accuracy** | 75.00% | 72.00% | 74.24% | **`89.39%`** |
| **Branch 1: Genuine Specificity** | 0.0% (0/33) | 69.7% (23/33) | 18.2% (6/33) | **`100.0%` (35/35, 0 false alarms)** |
| **Branch 1: Forged Recall** | 100.0% | 72.7% | 92.9% | **`85.6%` (83/97)** |
| **Branch 1: ROC-AUC** | 0.7815 | 0.7853 | 0.8606 | **`0.9446`** |
| **Branch 2: Raw Model Dice** | 0.0000 (Collapsed) | 0.0835 | 0.0988 | **`0.3330`** |
| **Branch 3: Categorization Accuracy** | 78.00% | 79.80% | 76.77% | **`82.47%`** |
| **Branch 3: Character Addition F1** | 0.9100 | 0.9100 | 0.8800 | **`1.0000` (Perfect 36/36)** |
| **Branch 3: Macro-Averaged F1** | 0.7789 | 0.7993 | 0.7666 | **`0.8005`** |

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/Aptee/ForgeryNet-Indic.git
cd ForgeryNet-Indic
pip install -r requirements.txt
```

### 2. Directory Structure
```
ForgeryNet/
├── src/
│   ├── dataset.py        # Native-resolution biased patch sampler & dataset loaders
│   ├── model.py          # ForgeryNet dual-stream architecture & spatial attention
│   ├── losses.py         # Multi-task loss (Balanced BCE + Focal Tversky + Masked CE)
│   ├── train.py          # Training loop with mixed precision (AMP) & scheduler
│   ├── evaluate.py       # Multi-task test metrics, ROC-AUC, Dice, Confusion Matrix
│   └── utils.py          # Visualizations, threshold calibration, and metrics
├── manuscript.tex        # LaTeX manuscript (Springer Nature / IJDAR format)
├── sn-bibliography.bib   # BibTeX references
├── ForgeryNet_Colab.ipynb# Full end-to-end interactive Google Colab notebook
├── DOCUMENTATION.md      # Detailed experimental trajectory and engineering report
└── requirements.txt      # Dependency specifications
```

### 3. Training
```bash
python src/train.py \
  --data_dir /path/to/Bengali_Handwriting_CopyMove_Dataset \
  --epochs 35 \
  --batch_size 8 \
  --lr 1.5e-4 \
  --patch_mode
```

### 4. Evaluation
```bash
python src/evaluate.py \
  --checkpoint outputs/checkpoints/best.pth \
  --data_dir /path/to/Bengali_Handwriting_CopyMove_Dataset \
  --patch_mode
```

---

## 📄 Citation
If you find this codebase or our methodology useful for your research, please cite our manuscript:

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
