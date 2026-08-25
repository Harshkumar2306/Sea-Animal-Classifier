# 🌊 Bio-HMSC+++ Model Suite

This directory contains the complete training, distributed computing, evaluation, and research paper toolchain for **Bio-HMSC+++** (*Biologically-Informed Hierarchical Multi-Scale Classifier*).

---

## 📁 File Structure & Overview

```
model/
├── model_architecture.py       # Modular PyTorch class definitions (BioHMSC, TaxonomyAwareLoss, Penalty Matrix)
├── train_ddp.py                # High-throughput Multi-GPU training script (PyTorch DDP / torchrun)
├── training1.py                # Single-GPU / DataParallel sequential 4-stage research pipeline
├── evaluate.py                 # Standalone checkpoint evaluator (produces 300 DPI plots & LaTeX table)
├── RESEARCH_PAPER_RESULTS.md   # Complete academic benchmark report, ablation tables & math formulation
└── README.md                   # Model suite documentation & usage guide
```

---

## 🚀 How to Train

### 1. Multi-GPU Distributed Training (Recommended for Dual T4 / A100 / V100)
Utilizes PyTorch's `torchrun` and NCCL backend to saturate all available GPUs symmetrically with zero CPU bottleneck:

```bash
torchrun --nproc_per_node=2 train_ddp.py
```

### 2. Single-GPU Training
Runs the full 4-stage training pipeline (Proposed Model + 4 Ablations + 3 Baselines + Visualizations) sequentially:

```bash
python training1.py
```

---

## 📊 How to Evaluate a Saved Checkpoint

To generate publication-grade figures (`confusion_matrix.png`, `roc_curves.png`) and LaTeX code from any saved checkpoint:

```bash
python evaluate.py --checkpoint proposed_best.pth --data_dir /path/to/split_dataset/test
```

### Evaluation Output Artifacts:
- **`confusion_matrix.png`**: High-resolution 300 DPI heatmap of fine-grained species classification.
- **`roc_curves.png`**: Publication-ready multi-class ROC-AUC curves for all 23 marine species.
- **`results_table.tex`**: Pre-formatted LaTeX table ready for direct inclusion into conference or journal manuscripts.

---

## 🔬 Core Methodological Innovations

1. **Hierarchical Multi-Task Learning:** Simultaneous prediction of fine-grained species ($C_f = 23$) and phylogenetic taxa ($C_c = 5$: *Mammalia, Aves, Pisces, Invertebrata, Reptilia*).
2. **Taxonomy-Aware Loss ($\mathcal{L}_{\text{tax}}$):** Mathematically penalizes cross-order classification errors $3\times$ more severely than intra-order confusions ($1.5$ penalty vs $0.5$ penalty).
3. **Weight Smoothing (EMA):** Exponential Moving Average ($\alpha = 0.9998$) for flatter loss landscape exploration.
4. **Group-Based Leakage-Free Splitting:** Groups scraped burst images by filename prefix to ensure zero overlap between train and test distributions.
