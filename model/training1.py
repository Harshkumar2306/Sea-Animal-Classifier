# -*- coding: utf-8 -*-
"""
Bio-HMSC+++ — KAGGLE VERSION
============================================
"""

import torch, timm, random, os, zipfile, shutil
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.utils as nn_utils
import numpy as np
import time

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from timm.utils import ModelEmaV2

from collections import Counter, defaultdict
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve,
                             top_k_accuracy_score)
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# REPRODUCIBILITY & DEVICE
# ============================================================

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

g = torch.Generator()
g.manual_seed(SEED)

# ============================================================
# HYPERPARAMETERS  (single source of truth)
# ============================================================

EPOCHS           = 40
num_gpus         = max(1, torch.cuda.device_count())
# Using 16 TOTAL batch size (8 per GPU on 2 GPUs) to avoid OOM, while maintaining effective BS of 32
BATCH_SIZE       = 16 
ACCUMULATION     = 2  
LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 1e-4
EMA_DECAY        = 0.9998
LABEL_SMOOTHING  = 0.1
CLIP_NORM        = 1.0
PATIENCE         = 10         # same for ALL models — no MIN_DELTA
TAX_LAMBDA       = 0.3
IMG_SIZE         = 384
DROP_RATE        = 0.5
DROP_PATH_RATE   = 0.2

# ============================================================
# 1. KAGGLE DATASET HANDLING (FULLY AUTOMATIC)
# ============================================================

CLEAN_DATA_ROOT = "/kaggle/working/cleaned_dataset_root"

if os.path.exists(CLEAN_DATA_ROOT) and len(os.listdir(CLEAN_DATA_ROOT)) > 0:
    print(f"Dataset already staged at '{CLEAN_DATA_ROOT}'. Skipping.")
else:
    print("Scanning /kaggle/input/ to find the dataset automatically...")
    
    best_root, max_subdirs = "/kaggle/input", 0
    
    # Dynamically search for the folder containing the most class subfolders
    for root, dirs, _ in os.walk("/kaggle/input"):
        valid = [d for d in dirs if not d.startswith('.') and not d.startswith('__')]
        if len(valid) > max_subdirs:
            max_subdirs, best_root = len(valid), root

    if max_subdirs == 0:
        raise FileNotFoundError("Could not find any class folders! Please check the right sidebar under 'Input' to ensure the dataset is actively attached to this notebook.")
        
    print(f"Found {max_subdirs} class folders at: {best_root}")
    print("Copying to working directory (this may take a minute)...")
    
    if os.path.exists(CLEAN_DATA_ROOT):
        shutil.rmtree(CLEAN_DATA_ROOT)
        
    shutil.copytree(best_root, CLEAN_DATA_ROOT, dirs_exist_ok=True)
    print("Copy complete.")

# ============================================================
# 2. DATA SPLIT  (70 / 15 / 15, group-based to prevent leakage)
# ============================================================

SPLIT_ROOT = "/kaggle/working/split_dataset"
TRAIN_DIR  = os.path.join(SPLIT_ROOT, "train")
VAL_DIR    = os.path.join(SPLIT_ROOT, "val")
TEST_DIR   = os.path.join(SPLIT_ROOT, "test")

if not os.path.exists(SPLIT_ROOT):
    print("Creating 70/15/15 group-based split...")
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    split_ratio = (0.70, 0.15, 0.15)

    for class_name in sorted(os.listdir(CLEAN_DATA_ROOT)):
        class_path = os.path.join(CLEAN_DATA_ROOT, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)

        # Group by filename prefix to avoid leaking near-duplicate images
        groups = defaultdict(list)
        for img in images:
            parts = img.replace('-', '_').split('_')
            prefix = "_".join(parts[:2]) if len(parts) > 1 else img
            groups[prefix].append(img)

        group_keys = list(groups.keys())
        random.shuffle(group_keys)

        n = len(images)
        train_imgs, val_imgs, test_imgs = [], [], []
        for key in group_keys:
            if len(train_imgs) < n * split_ratio[0]:
                train_imgs.extend(groups[key])
            elif len(val_imgs) < n * split_ratio[1]:
                val_imgs.extend(groups[key])
            else:
                test_imgs.extend(groups[key])

        for folder, img_list in zip(
            [TRAIN_DIR, VAL_DIR, TEST_DIR],
            [train_imgs, val_imgs, test_imgs]
        ):
            dest = os.path.join(folder, class_name)
            os.makedirs(dest, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(class_path, img), os.path.join(dest, img))

    print("Split complete.")
else:
    print(f"Split already exists at '{SPLIT_ROOT}'.")

# ============================================================
# 3. CLASS DEFINITIONS & TAXONOMY
# ============================================================

temp_ds      = datasets.ImageFolder(TRAIN_DIR)
fine_classes = temp_ds.classes
num_classes  = len(fine_classes)
print(f"Loaded {num_classes} fine-grained classes.")

COARSE_MAP = {
    "whale": "mammal",   "dolphin": "mammal", "seal": "mammal",
    "otter": "mammal",   "sea otter": "mammal",
    "penguin": "bird",
    "fish": "fish",      "puffers": "fish",   "sea rays": "fish",
    "eel": "fish",       "seahorse": "fish",  "sharks": "fish",
    "octopus": "invertebrate",  "squid": "invertebrate",
    "jelly fish": "invertebrate", "starfish": "invertebrate",
    "lobster": "invertebrate",  "shrimp": "invertebrate",
    "crabs": "invertebrate",    "corals": "invertebrate",
    "sea urchins": "invertebrate", "clams": "invertebrate",
    "nudibranchs": "invertebrate",
    "turtle tortoise": "reptile",
}

coarse_classes = sorted(set(COARSE_MAP[c.lower().replace('_', ' ')] for c in fine_classes))
coarse_to_idx  = {c: i for i, c in enumerate(coarse_classes)}
num_coarse     = len(coarse_classes)

# Penalty matrix: same coarse group → 0.5, different group → 1.5
penalty_matrix = torch.ones(num_classes, num_classes, device=device)
for i, ci in enumerate(fine_classes):
    for j, cj in enumerate(fine_classes):
        if COARSE_MAP[ci.lower().replace('_', ' ')] == COARSE_MAP[cj.lower().replace('_', ' ')]:
            penalty_matrix[i, j] = 0.5

# ============================================================
# 4. TRANSFORMS
# ============================================================

# High-throughput marine augmentations (10x faster than AutoAugment on CPU)
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================
# 5. DATASET CLASS
# ============================================================

class MarineDataset(Dataset):
    """Wraps ImageFolder to return (image, fine_label, coarse_label)."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, fine_idx = self.base[idx]
        coarse_idx = coarse_to_idx[COARSE_MAP[fine_classes[fine_idx].lower().replace('_', ' ')]]
        return img, fine_idx, coarse_idx

# ============================================================
# 6. DATALOADERS
# ============================================================

train_base = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
val_base   = datasets.ImageFolder(VAL_DIR,   transform=val_tfms)
test_base  = datasets.ImageFolder(TEST_DIR,  transform=val_tfms)

train_dataset = MarineDataset(train_base)
val_dataset   = MarineDataset(val_base)
test_dataset  = MarineDataset(test_base)

print(f"\nDataset split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
print("Train class distribution:", dict(Counter(train_base.targets)))

targets        = train_base.targets
count          = Counter(targets)
class_weights  = 1.0 / torch.tensor([count[i] for i in range(num_classes)], dtype=torch.float)
sample_weights = [class_weights[t] for t in targets]
sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), generator=g)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2, generator=g)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

# ============================================================
# 7. MODEL ARCHITECTURE
# ============================================================

class BioHMSC(nn.Module):
    def __init__(self, backbone_name: str = "tf_efficientnetv2_m"):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0,
            drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE
        )
        feat_dim = self.backbone.num_features
        self.shared_mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(DROP_RATE),
        )
        self.species_head = nn.Linear(512, num_classes)
        self.coarse_head  = nn.Linear(512, num_coarse)

    def forward(self, x):
        feats        = self.backbone(x)
        shared_feats = self.shared_mlp(feats)
        return self.species_head(shared_feats), self.coarse_head(shared_feats)

# ============================================================
# 8. LOSS FUNCTIONS
# ============================================================

def taxonomy_aware_loss(logits, targets):
    ce           = nn.functional.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING)
    probs        = torch.softmax(logits, dim=1)
    penalties    = penalty_matrix[targets]          # (B, C)
    penalty_term = torch.sum(probs * penalties, dim=1).mean()
    return ce + TAX_LAMBDA * penalty_term

def standard_ce_loss(logits, targets):
    return nn.functional.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING)

# ============================================================
# 9. CORE TRAINING & EVALUATION HELPERS
# ============================================================

def train_one_epoch(model, ema, loader, optimizer, scaler, loss_fn, use_coarse_loss=True):
    model.train()
    optimizer.zero_grad() 
    running_loss = correct = total = 0

    for batch_idx, (imgs, fine_lbl, coarse_lbl) in enumerate(loader):
        imgs      = imgs.to(device, non_blocking=True)
        fine_lbl  = fine_lbl.to(device, non_blocking=True)
        coarse_lbl = coarse_lbl.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            s_out, c_out = model(imgs)
            loss = loss_fn(s_out, fine_lbl)
            if use_coarse_loss:
                loss = loss + nn.functional.cross_entropy(c_out, coarse_lbl)
            loss = loss / ACCUMULATION

        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        running_loss += loss.item() * ACCUMULATION
        correct      += (s_out.argmax(1) == fine_lbl).sum().item()
        total        += fine_lbl.size(0)

    return running_loss / len(loader), 100.0 * correct / total

def validate(eval_model, loader):
    eval_model.eval()
    running_loss = correct = total = 0
    with torch.no_grad():
        for imgs, fine_lbl, coarse_lbl in loader:
            imgs       = imgs.to(device, non_blocking=True)
            fine_lbl   = fine_lbl.to(device, non_blocking=True)
            coarse_lbl = coarse_lbl.to(device, non_blocking=True)
            out, c_out = eval_model(imgs)
            loss       = taxonomy_aware_loss(out, fine_lbl) \
                       + nn.functional.cross_entropy(c_out, coarse_lbl)
            running_loss += loss.item()
            correct      += (out.argmax(1) == fine_lbl).sum().item()
            total        += fine_lbl.size(0)
    return running_loss / len(loader), 100.0 * correct / total

def evaluate_full(eval_model, loader, apply_tta=False):
    eval_model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            if apply_tta:
                out1, _ = eval_model(imgs)
                out2, _ = eval_model(torch.flip(imgs, dims=[3]))
                outputs  = (out1 + out2) / 2.0
            else:
                outputs, _ = eval_model(imgs)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return (np.array(all_preds), np.array(all_labels), np.array(all_probs))

def compute_metrics(preds, labels, probs, label="Model"):
    acc        = accuracy_score(labels, preds) * 100
    macro_f1   = f1_score(labels, preds, average="macro",    zero_division=0) * 100
    weighted_f1= f1_score(labels, preds, average="weighted", zero_division=0) * 100
    top3       = top_k_accuracy_score(labels, probs, k=3)   * 100
    top5       = top_k_accuracy_score(labels, probs, k=5)   * 100
    try:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr")
    except Exception:
        roc_auc = float("nan")

    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  Accuracy        : {acc:.2f}%")
    print(f"  Macro F1        : {macro_f1:.2f}%")
    print(f"  Weighted F1     : {weighted_f1:.2f}%")
    print(f"  Top-3 Accuracy  : {top3:.2f}%")
    print(f"  Top-5 Accuracy  : {top5:.2f}%")
    print(f"  ROC-AUC (OvR)   : {roc_auc:.4f}")
    return dict(acc=acc, macro_f1=macro_f1, weighted_f1=weighted_f1,
                top3=top3, top5=top5, roc_auc=roc_auc)

def full_train_run(
    backbone_name,
    loss_fn,
    use_coarse_loss,
    use_ema,
    model_save_path,
    patience=PATIENCE,
    label="",
):
    print(f"\n{'#'*60}")
    print(f"  TRAINING: {label}")
    print(f"  backbone={backbone_name}, tax_loss={loss_fn.__name__}, "
          f"coarse={use_coarse_loss}, ema={use_ema}")
    print(f"{'#'*60}")

    model = BioHMSC(backbone_name).to(device)
    if torch.cuda.device_count() > 1:
        print(f"  🔥 Utilizing {torch.cuda.device_count()} GPUs via DataParallel!")
        model = nn.DataParallel(model)
        
    ema   = ModelEmaV2(model, decay=EMA_DECAY) if use_ema else None

    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1,
                                             eta_min=1e-6)
    scaler     = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []

    best_val_acc    = 0.0
    best_epoch      = 0
    early_stop_ctr  = 0
    run_start       = time.time()
    
    # Save models directly to /kaggle/working/
    full_save_path = f"/kaggle/working/{model_save_path}"

    for epoch in range(EPOCHS):
        ep_start = time.time()

        t_loss, t_acc = train_one_epoch(
            model, ema, train_loader, optimizer, scaler,
            loss_fn, use_coarse_loss=use_coarse_loss
        )

        eval_model = ema.module if use_ema else model
        v_loss, v_acc = validate(eval_model, val_loader)

        scheduler.step()

        train_loss_hist.append(t_loss)
        val_loss_hist.append(v_loss)
        train_acc_hist.append(t_acc)
        val_acc_hist.append(v_acc)

        elapsed = time.time() - ep_start
        print(f"  Ep {epoch+1:02d}/{EPOCHS} [{elapsed:.0f}s] "
              f"train_acc={t_acc:.2f}% | val_acc={v_acc:.2f}% "
              f"| val_loss={v_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if v_acc > best_val_acc:
            best_val_acc    = v_acc
            best_epoch      = epoch + 1
            early_stop_ctr  = 0
            
            # Strip DataParallel wrapper before saving to avoid 'module.' prefix
            model_to_save = eval_model.module if isinstance(eval_model, nn.DataParallel) else eval_model
            torch.save(model_to_save.state_dict(), full_save_path)
            
            print(f"    ✓ Best model saved (val_acc={v_acc:.2f}%)")
        else:
            early_stop_ctr += 1
            if early_stop_ctr >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience}).")
                break

    total_mins = (time.time() - run_start) / 60.0
    print(f"\n  Best val acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"  Total training time: {total_mins:.1f} min  "
          f"({total_mins*60/max(1,len(train_loss_hist)):.0f} s/epoch)")

    eval_model = BioHMSC(backbone_name).to(device)
    eval_model.load_state_dict(torch.load(full_save_path, weights_only=True))

    return eval_model, {
        "train_loss": train_loss_hist,
        "val_loss":   val_loss_hist,
        "train_acc":  train_acc_hist,
        "val_acc":    val_acc_hist,
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch,
        "total_mins":   total_mins,
    }


# ============================================================
# STAGE 1 — PROPOSED MODEL 
# ============================================================

print("\n" + "="*60)
print("STAGE 1: PROPOSED MODEL — Bio-HMSC+++ (EfficientNetV2-M)")
print("="*60)

proposed_model, proposed_hist = full_train_run(
    backbone_name    = "tf_efficientnetv2_m",
    loss_fn          = taxonomy_aware_loss,
    use_coarse_loss  = True,
    use_ema          = True,
    model_save_path  = "proposed_best.pth",
    label            = "Bio-HMSC+++ — EfficientNetV2-M + Tax Loss + EMA",
)

preds_prop, labels_prop, probs_prop = evaluate_full(
    proposed_model, test_loader, apply_tta=True
)
metrics_proposed = compute_metrics(
    preds_prop, labels_prop, probs_prop,
    label="Bio-HMSC+++ (EfficientNetV2-M + TaxLoss + EMA + TTA)"
)

coarse_preds_list, coarse_labels_list = [], []
proposed_model.eval()
with torch.no_grad():
    for imgs, _, coarse_lbl in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        _, co1 = proposed_model(imgs)
        _, co2 = proposed_model(torch.flip(imgs, dims=[3]))
        c_avg  = (co1 + co2) / 2.0
        coarse_preds_list.extend(c_avg.argmax(1).cpu().numpy())
        coarse_labels_list.extend(coarse_lbl.numpy())
coarse_acc = accuracy_score(coarse_labels_list, coarse_preds_list) * 100
print(f"\n  Coarse-level Accuracy (Proposed): {coarse_acc:.2f}%")


# ============================================================
# STAGE 2 — ABLATION STUDY
# ============================================================

print("\n" + "="*60)
print("STAGE 2: PROPER 4-CONFIGURATION ABLATION STUDY")
print("="*60)

model_A, hist_A = full_train_run(
    backbone_name   = "tf_efficientnetv2_m",
    loss_fn         = standard_ce_loss,
    use_coarse_loss = False,
    use_ema         = False,
    model_save_path = "ablation_A.pth",
    label           = "Config A: EfficientNetV2-M + Standard CE (no EMA, no TTA)",
)
preds_A, labels_A, probs_A = evaluate_full(model_A, test_loader, apply_tta=False)
metrics_A = compute_metrics(preds_A, labels_A, probs_A, label="Config A")

model_B, hist_B = full_train_run(
    backbone_name   = "tf_efficientnetv2_m",
    loss_fn         = taxonomy_aware_loss,
    use_coarse_loss = True,
    use_ema         = False,
    model_save_path = "ablation_B.pth",
    label           = "Config B: + Taxonomy Loss (no EMA, no TTA)",
)
preds_B, labels_B, probs_B = evaluate_full(model_B, test_loader, apply_tta=False)
metrics_B = compute_metrics(preds_B, labels_B, probs_B, label="Config B")

model_C, hist_C = full_train_run(
    backbone_name   = "tf_efficientnetv2_m",
    loss_fn         = taxonomy_aware_loss,
    use_coarse_loss = True,
    use_ema         = True,
    model_save_path = "ablation_C.pth",
    label           = "Config C: + EMA (no TTA)",
)
preds_C, labels_C, probs_C = evaluate_full(model_C, test_loader, apply_tta=False)
metrics_C = compute_metrics(preds_C, labels_C, probs_C, label="Config C")

preds_D_noTTA, _, probs_D_noTTA = evaluate_full(proposed_model, test_loader, apply_tta=False)
metrics_D_noTTA = compute_metrics(preds_D_noTTA, labels_prop, probs_D_noTTA,
                                  label="Config D: Full Bio-HMSC+++ (no TTA, for ablation table)")
metrics_D_TTA = metrics_proposed

print("\n\n--- ABLATION SUMMARY ---")
print(f"  Config A (CE only, no EMA, no TTA)  : {metrics_A['acc']:.2f}%")
print(f"  Config B (+ TaxLoss, no EMA, no TTA): {metrics_B['acc']:.2f}%  "
      f"[Δ vs A: {metrics_B['acc']-metrics_A['acc']:+.2f}pp]")
print(f"  Config C (+ EMA, no TTA)            : {metrics_C['acc']:.2f}%  "
      f"[Δ vs B: {metrics_C['acc']-metrics_B['acc']:+.2f}pp]")
print(f"  Config D (+ TTA)                    : {metrics_D_noTTA['acc']:.2f}%  "
      f"[Δ vs C: {metrics_D_noTTA['acc']-metrics_C['acc']:+.2f}pp]")
print(f"  Config D (final, WITH TTA)           : {metrics_D_TTA['acc']:.2f}%")


# ============================================================
# STAGE 3 — BASELINE COMPARISONS
# ============================================================

print("\n" + "="*60)
print("STAGE 3: BASELINE COMPARISONS")
print("="*60)

BASELINES = [
    ("mobilenetv2_100", "MobileNetV2"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("resnet50",        "ResNet50"),
]

baseline_results = {}

for b_id, b_label in BASELINES:
    b_model, b_hist = full_train_run(
        backbone_name   = b_id,
        loss_fn         = standard_ce_loss, 
        use_coarse_loss = False,
        use_ema         = False,
        model_save_path = f"{b_id}_baseline.pth",
        label           = f"Baseline: {b_label}",
    )
    b_preds, b_labels, b_probs = evaluate_full(b_model, test_loader, apply_tta=True)
    b_metrics = compute_metrics(b_preds, b_labels, b_probs, label=f"{b_label} + TTA")
    baseline_results[b_label] = {
        "model":   b_model,
        "metrics": b_metrics,
        "hist":    b_hist,
    }


# ============================================================
# STAGE 4 — PAPER VISUALIZATIONS & METRICS REPORT
# ============================================================

print("\n" + "*"*60)
print("STAGE 4: GENERATING PAPER FEATURES & VISUALIZATIONS")
print("*"*60)

# All plots and charts will automatically save to /kaggle/working/
print("\n[1] Confusion Matrix (Proposed Model, Test Set)")
cm = confusion_matrix(labels_prop, preds_prop)
plt.figure(figsize=(15, 13))
sns.heatmap(cm, annot=False, cmap='Blues',
            xticklabels=fine_classes, yticklabels=fine_classes)
plt.title("Confusion Matrix — Bio-HMSC+++ (Test Set, TTA)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0,  fontsize=8)
plt.tight_layout()
plt.savefig("/kaggle/working/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n[2] Classification Report (Proposed Model):")
print(classification_report(labels_prop, preds_prop,
                             target_names=fine_classes, zero_division=0))

print("\n[3] Training Curves (Proposed Model)")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(proposed_hist["train_loss"], label="Train Loss",  color="blue",  lw=2)
axes[0].plot(proposed_hist["val_loss"],   label="Val Loss",    color="red",   lw=2)
axes[0].set_title("Training & Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)
axes[1].plot(proposed_hist["train_acc"],  label="Train Accuracy", color="blue", lw=2)
axes[1].plot(proposed_hist["val_acc"],    label="Val Accuracy",   color="red",  lw=2)
axes[1].set_title("Training & Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig("/kaggle/working/training_curves.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n[4] Precision & Recall (Proposed Model, Weighted):")
prec = precision_score(labels_prop, preds_prop, average="weighted", zero_division=0)
rec  = recall_score(labels_prop,    preds_prop, average="weighted", zero_division=0)
print(f"  Weighted Precision : {prec:.4f}")
print(f"  Weighted Recall    : {rec:.4f}")

print("\n[5] Training Complexity:")
print(f"  GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  Bio-HMSC+++ — {proposed_hist['total_mins']:.1f} min total  "
      f"| {proposed_hist['total_mins']*60/len(proposed_hist['train_loss']):.0f} s/epoch  "
      f"| ~{sum(p.numel() for p in proposed_model.parameters())/1e6:.1f}M params")
for b_label, b_data in baseline_results.items():
    h = b_data["hist"]
    m = b_data["model"]
    print(f"  {b_label:20s} — {h['total_mins']:.1f} min total  "
          f"| {h['total_mins']*60/len(h['train_loss']):.0f} s/epoch  "
          f"| ~{sum(p.numel() for p in m.parameters())/1e6:.1f}M params")

print("\n[6] ABLATION STUDY TABLE (Empirical — All Values Measured):")
print(f"  {'Configuration':<48} {'Accuracy':>9} {'Macro F1':>9} {'Δ Acc':>7}")
print(f"  {'-'*75}")
configs = [
    ("Config A: EffNetV2-M + Standard CE             ", metrics_A['acc'],    metrics_A['macro_f1'],    None),
    ("Config B: + Taxonomy-Aware Loss                ", metrics_B['acc'],    metrics_B['macro_f1'],    metrics_B['acc'] - metrics_A['acc']),
    ("Config C: + EMA Weight Smoothing               ", metrics_C['acc'],    metrics_C['macro_f1'],    metrics_C['acc'] - metrics_B['acc']),
    ("Config D: + TTA (Full Bio-HMSC+++)             ", metrics_D_TTA['acc'],metrics_D_TTA['macro_f1'],metrics_D_TTA['acc'] - metrics_C['acc']),
]
for name, acc, mf1, delta in configs:
    d_str = f"{delta:+.2f}pp" if delta is not None else "  —   "
    print(f"  {name} {acc:>8.2f}%  {mf1:>8.2f}%  {d_str:>7}")

print("\n[7] COMPARISON TABLE (All Models Evaluated WITH TTA — Fair Comparison):")
print(f"  {'Model':<25} {'Accuracy':>9} {'Macro F1':>9} {'ROC-AUC':>9} {'Params':>9}")
print(f"  {'-'*65}")
for b_label, b_data in baseline_results.items():
    bm = b_data["metrics"]
    pm = b_data["model"]
    params = sum(p.numel() for p in pm.parameters()) / 1e6
    print(f"  {b_label:<25} {bm['acc']:>8.2f}%  {bm['macro_f1']:>8.2f}%  "
          f"{bm['roc_auc']:>8.4f}  {params:>6.1f}M")
prop_params = sum(p.numel() for p in proposed_model.parameters()) / 1e6
print(f"  {'Bio-HMSC+++ (Proposed)':<25} {metrics_proposed['acc']:>8.2f}%  "
      f"{metrics_proposed['macro_f1']:>8.2f}%  "
      f"{metrics_proposed['roc_auc']:>8.4f}  {prop_params:>6.1f}M")

print("\n[8] ROC Curves (All 23 Marine Classes)...")
try:
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(labels_prop, classes=range(num_classes))
    plt.figure(figsize=(13, 9))
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_prop[:, i])
        plt.plot(fpr, tpr, label=fine_classes[i], color=colors[i], lw=1.2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random")
    plt.title("ROC Curves — Bio-HMSC+++ (All 23 Classes)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.savefig("/kaggle/working/roc_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
except Exception as exc:
    print(f"  ROC plot error: {exc}")

print("\n[9] Dataset Statistics:")
print(f"  Total images   : {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
print(f"  Train set      : {len(train_dataset):>5} images  (70%)")
print(f"  Validation set : {len(val_dataset):>5} images  (15%)")
print(f"  Test set       : {len(test_dataset):>5} images  (15%)")
print(f"  Fine-grained classes : {num_classes}")
print(f"  Coarse categories    : {num_coarse} ({', '.join(coarse_classes)})")
print(f"  Class imbalance      : handled via WeightedRandomSampler (training only)")
print(f"  Splitting strategy   : group-based (filename prefix) to prevent data leakage")

print("\n[10] Error Analysis:")
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)
err_indices = np.dstack(np.unravel_index(
    np.argsort(cm_copy, axis=None)[::-1][:5], cm_copy.shape
))[0]
print("  Top-5 most frequent confusions (proposed model):")
for true_idx, pred_idx in err_indices:
    if cm_copy[true_idx, pred_idx] > 0:
        print(f"    '{fine_classes[true_idx]}' → '{fine_classes[pred_idx]}': "
              f"{cm_copy[true_idx, pred_idx]} times")

print("\n\n" + "="*60)
print("ALL STAGES COMPLETE.")
print("="*60)
print(f"\nFinal proposed model test accuracy (with TTA): {metrics_proposed['acc']:.2f}%")
print(f"Final proposed model macro F1 (with TTA)     : {metrics_proposed['macro_f1']:.2f}%")
print(f"Final proposed model ROC-AUC (OvR)           : {metrics_proposed['roc_auc']:.4f}")
print(f"Coarse-level accuracy                        : {coarse_acc:.2f}%")