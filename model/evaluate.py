# -*- coding: utf-8 -*-
"""
Bio-HMSC+++ Standalone Evaluation & Publication Diagnostics
===========================================================
Evaluates any saved checkpoint (.pth), generates high-resolution figures (DPI 300),
produces full classification metrics, and exports LaTeX code for research papers.

Usage:
    python evaluate.py --checkpoint proposed_best.pth --data_dir /kaggle/working/split_dataset/test
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from model_architecture import BioHMSC, FINE_CLASSES, COARSE_MAP, COARSE_CLASSES, COARSE_TO_IDX

# Set styling for IEEE / Springer publication-grade plots
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'savefig.dpi': 300,
})

class MarineDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.fine_classes = base_dataset.classes
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, fine_idx = self.base[idx]
        c_name = self.fine_classes[fine_idx].lower().replace('_', ' ')
        coarse_idx = COARSE_TO_IDX.get(COARSE_MAP.get(c_name, "invertebrate"), 0)
        return img, fine_idx, coarse_idx

def run_evaluation(checkpoint_path, test_dir, backbone_name="tf_efficientnetv2_m", output_dir="eval_results", apply_tta=True):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  Bio-HMSC+++ Research Evaluation Engine")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device} | TTA: {apply_tta}")
    print(f"=======================================================\n")

    # Load Model
    model = BioHMSC(backbone_name=backbone_name, num_fine_classes=len(FINE_CLASSES), num_coarse_classes=len(COARSE_CLASSES)).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Strip any possible DDP prefix if present
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state)
    model.eval()

    # Transforms
    test_tfms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_base = datasets.ImageFolder(test_dir, transform=test_tfms)
    test_dataset = MarineDataset(test_base)
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    all_preds, all_labels, all_probs = [], [], []
    coarse_preds, coarse_labels = [], []

    print("Running inference across test partition...")
    with torch.no_grad():
        for imgs, fine_lbl, coarse_lbl in loader:
            imgs = imgs.to(device, non_blocking=True)
            if apply_tta:
                out1, c1 = model(imgs)
                out2, c2 = model(torch.flip(imgs, dims=[3]))
                outputs = (out1 + out2) / 2.0
                c_outputs = (c1 + c2) / 2.0
            else:
                outputs, c_outputs = model(imgs)

            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(fine_lbl.numpy())
            all_probs.extend(probs.cpu().numpy())

            coarse_preds.extend(c_outputs.argmax(1).cpu().numpy())
            coarse_labels.extend(coarse_lbl.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    c_preds = np.array(coarse_preds)
    c_labels = np.array(coarse_labels)

    # 1. CORE QUANTITATIVE METRICS
    acc = accuracy_score(labels, preds) * 100
    top3 = top_k_accuracy_score(labels, probs, k=3) * 100
    top5 = top_k_accuracy_score(labels, probs, k=5) * 100
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0) * 100
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0) * 100
    weighted_prec = precision_score(labels, preds, average="weighted", zero_division=0) * 100
    weighted_rec = recall_score(labels, preds, average="weighted", zero_division=0) * 100
    coarse_acc = accuracy_score(c_labels, c_preds) * 100

    try:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr")
    except Exception:
        roc_auc = float("nan")

    print("\n-------------------------------------------------------")
    print("  SUMMARY EVALUATION METRICS:")
    print("-------------------------------------------------------")
    print(f"  • Top-1 Accuracy            : {acc:.2f}%")
    print(f"  • Top-3 Accuracy            : {top3:.2f}%")
    print(f"  • Top-5 Accuracy            : {top5:.2f}%")
    print(f"  • Macro F1-Score            : {macro_f1:.2f}%")
    print(f"  • Weighted F1-Score         : {weighted_f1:.2f}%")
    print(f"  • Weighted Precision        : {weighted_prec:.2f}%")
    print(f"  • Weighted Recall           : {weighted_rec:.2f}%")
    print(f"  • Multi-class ROC-AUC (OvR) : {roc_auc:.4f}")
    print(f"  • Coarse Taxa Accuracy      : {coarse_acc:.2f}%")
    print("-------------------------------------------------------\n")

    # 2. GENERATE CONFUSION MATRIX
    print("Exporting Publication-Quality Confusion Matrix...")
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=test_base.classes, yticklabels=test_base.classes)
    plt.title(f"Bio-HMSC+++ Confusion Matrix (Test Acc: {acc:.2f}%)", pad=15)
    plt.xlabel("Predicted Taxonomic Species", labelpad=10)
    plt.ylabel("True Ground Truth Species", labelpad=10)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. GENERATE MULTI-CLASS ROC CURVES
    print("Exporting Publication-Quality ROC-AUC Curves...")
    try:
        y_bin = label_binarize(labels, classes=range(len(test_base.classes)))
        plt.figure(figsize=(13, 9))
        colors = plt.cm.tab20(np.linspace(0, 1, len(test_base.classes)))
        for i, c_name in enumerate(test_base.classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f"{c_name}", color=colors[i], lw=1.3)
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (AUC=0.50)")
        plt.title(f"Multi-Class ROC Curves — Bio-HMSC+++ (Macro ROC-AUC: {roc_auc:.4f})", pad=15)
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8, ncol=2)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, "roc_curves.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  Warning: ROC Plot generation error: {e}")

    # 4. GENERATE LATEX TABLE
    latex_code = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Quantitative Performance Summary of Proposed Bio-HMSC+++ Architecture}}
\\label{{tab:biolmsc_results}}
\\begin{{tabular}}{{lccccc}}
\\hline
\\textbf{{Model Architecture}} & \\textbf{{Top-1 Acc (\\%)}} & \\textbf{{Top-5 Acc (\\%)}} & \\textbf{{Macro F1 (\\%)}} & \\textbf{{ROC-AUC}} & \\textbf{{Coarse Acc (\\%)}}\\\\
\\hline
Bio-HMSC+++ (Proposed) & {acc:.2f} & {top5:.2f} & {macro_f1:.2f} & {roc_auc:.4f} & {coarse_acc:.2f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    latex_file = os.path.join(output_dir, "results_table.tex")
    with open(latex_file, "w") as f:
        f.write(latex_code.strip())

    print(f"All research paper artifacts exported to folder: {output_dir}/")
    print("✓ confusion_matrix.png (300 DPI)")
    print("✓ roc_curves.png (300 DPI)")
    print("✓ results_table.tex (LaTeX Table)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bio-HMSC+++ Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="/kaggle/working/proposed_best.pth", help="Path to .pth checkpoint")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/split_dataset/test", help="Path to test image folder")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save figures")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        test_dir=args.data_dir,
        output_dir=args.output_dir,
        apply_tta=(not args.no_tta)
    )
