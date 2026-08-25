# -*- coding: utf-8 -*-
"""
Bio-HMSC+++ — DISTRIBUTED DATA PARALLEL (DDP) FULL PAPER PIPELINE
================================================================
Paper-Ready Multi-Stage Training Pipeline:
  • STAGE 1: Proposed Bio-HMSC+++ (EfficientNetV2-M + TaxLoss + EMA + TTA)
  • STAGE 2: 4-Configuration Ablation Study (Empirical Measurement)
  • STAGE 3: Baselines Comparison (MobileNetV2, EfficientNet-B0, ResNet50)
  • STAGE 4: Publication Visualizations (CM, ROC-AUC, Dual Curves, LaTeX Table)

Run command:
  torchrun --nproc_per_node=2 train_ddp.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, timm, random, shutil, time
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.utils as nn_utils
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from timm.utils import ModelEmaV2

from collections import Counter, defaultdict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve,
                             top_k_accuracy_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. DDP CLUSTER SETUP & INITIALIZATION
# ============================================================

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

rank, world_size, local_rank = setup_ddp()
is_main_process = (rank == 0)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

SEED = 42 + rank
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

def print_main(*args, **kwargs):
    if is_main_process:
        print(*args, **kwargs)

print_main(f"\n[DDP System] Initialized {world_size} GPUs (Local Rank: {local_rank}) | cuDNN Autotuner Active")

# ============================================================
# 2. HYPERPARAMETERS
# ============================================================

EPOCHS             = 40
BATCH_SIZE_PER_GPU = 16        # 16 per GPU = 32 Global Batch Size (Optimal memory stability)
GLOBAL_BATCH_SIZE  = BATCH_SIZE_PER_GPU * world_size
ACCUMULATION       = 1
LEARNING_RATE      = 1.2e-4
WEIGHT_DECAY       = 1e-4
EMA_DECAY          = 0.9998
LABEL_SMOOTHING    = 0.1
CLIP_NORM          = 1.0
PATIENCE           = 10
TAX_LAMBDA         = 0.3
IMG_SIZE           = 384
DROP_RATE          = 0.5
DROP_PATH_RATE     = 0.2

# ============================================================
# 3. KAGGLE DATASET AUTOMATIC STAGING (RANK 0 ONLY)
# ============================================================

CLEAN_DATA_ROOT = "/kaggle/working/cleaned_dataset_root"
SPLIT_ROOT      = "/kaggle/working/split_dataset"
TRAIN_DIR       = os.path.join(SPLIT_ROOT, "train")
VAL_DIR         = os.path.join(SPLIT_ROOT, "val")
TEST_DIR        = os.path.join(SPLIT_ROOT, "test")

if is_main_process:
    if not os.path.exists(CLEAN_DATA_ROOT) or len(os.listdir(CLEAN_DATA_ROOT)) == 0:
        print_main("Scanning /kaggle/input/ for Sea Animals dataset...")
        best_root, max_subdirs = "/kaggle/input", 0
        for root, dirs, _ in os.walk("/kaggle/input"):
            valid = [d for d in dirs if not d.startswith('.') and not d.startswith('__')]
            if len(valid) > max_subdirs:
                max_subdirs, best_root = len(valid), root
        if max_subdirs == 0:
            raise FileNotFoundError("Could not find any class folders in /kaggle/input/!")
        shutil.copytree(best_root, CLEAN_DATA_ROOT, dirs_exist_ok=True)

    if not os.path.exists(SPLIT_ROOT):
        print_main("Creating 70/15/15 group-based split (zero data leakage)...")
        for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            os.makedirs(d, exist_ok=True)
        split_ratio = (0.70, 0.15, 0.15)
        for class_name in sorted(os.listdir(CLEAN_DATA_ROOT)):
            class_path = os.path.join(CLEAN_DATA_ROOT, class_name)
            if not os.path.isdir(class_path): continue
            images = os.listdir(class_path)
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
                if len(train_imgs) < n * split_ratio[0]: train_imgs.extend(groups[key])
                elif len(val_imgs) < n * split_ratio[1]: val_imgs.extend(groups[key])
                else: test_imgs.extend(groups[key])
            for folder, img_list in zip([TRAIN_DIR, VAL_DIR, TEST_DIR], [train_imgs, val_imgs, test_imgs]):
                dest = os.path.join(folder, class_name)
                os.makedirs(dest, exist_ok=True)
                for img in img_list:
                    shutil.copy(os.path.join(class_path, img), os.path.join(dest, img))
        print_main("Split complete.")

if world_size > 1:
    dist.barrier()

# ============================================================
# 4. CLASS DEFINITIONS & TAXONOMY
# ============================================================

temp_ds      = datasets.ImageFolder(TRAIN_DIR)
fine_classes = temp_ds.classes
num_classes  = len(fine_classes)

COARSE_MAP = {
    "whale": "mammal",           "dolphin": "mammal",         "seal": "mammal",
    "otter": "mammal",           "sea otter": "mammal",
    "penguin": "bird",
    "fish": "fish",              "puffers": "fish",           "sea rays": "fish",
    "eel": "fish",               "seahorse": "fish",          "sharks": "fish",
    "octopus": "invertebrate",   "squid": "invertebrate",     "jelly fish": "invertebrate",
    "starfish": "invertebrate",  "lobster": "invertebrate",   "shrimp": "invertebrate",
    "crabs": "invertebrate",     "corals": "invertebrate",    "sea urchins": "invertebrate",
    "clams": "invertebrate",     "nudibranchs": "invertebrate",
    "turtle tortoise": "reptile"
}

coarse_classes = sorted(set(COARSE_MAP[c.lower().replace('_', ' ')] for c in fine_classes))
coarse_to_idx  = {c: i for i, c in enumerate(coarse_classes)}
num_coarse     = len(coarse_classes)

penalty_matrix = torch.ones(num_classes, num_classes, device=device)
for i, ci in enumerate(fine_classes):
    for j, cj in enumerate(fine_classes):
        if COARSE_MAP[ci.lower().replace('_', ' ')] == COARSE_MAP[cj.lower().replace('_', ' ')]:
            penalty_matrix[i, j] = 0.5

# ============================================================
# 5. DATA LOADERS
# ============================================================

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

class MarineDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, fine_idx = self.base[idx]
        coarse_idx = coarse_to_idx[COARSE_MAP[fine_classes[fine_idx].lower().replace('_', ' ')]]
        return img, fine_idx, coarse_idx

train_base = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
val_base   = datasets.ImageFolder(VAL_DIR,   transform=val_tfms)
test_base  = datasets.ImageFolder(TEST_DIR,  transform=val_tfms)

train_dataset = MarineDataset(train_base)
val_dataset   = MarineDataset(val_base)
test_dataset  = MarineDataset(test_base)

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
val_sampler   = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler,
                          shuffle=(train_sampler is None), num_workers=4, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=val_sampler,
                          shuffle=False, num_workers=4, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE_PER_GPU, shuffle=False,
                          num_workers=4, pin_memory=True)

# ============================================================
# 6. MODEL ARCHITECTURE
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
        feats = self.backbone(x)
        shared = self.shared_mlp(feats)
        return self.species_head(shared), self.coarse_head(shared)

def taxonomy_aware_loss(logits, targets):
    ce           = nn.functional.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING)
    probs        = torch.softmax(logits, dim=1)
    penalties    = penalty_matrix[targets]
    penalty_term = torch.sum(probs * penalties, dim=1).mean()
    return ce + TAX_LAMBDA * penalty_term

def standard_ce_loss(logits, targets):
    return nn.functional.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING)

# ============================================================
# 7. TRAINING & EVALUATION HELPERS
# ============================================================

def train_one_epoch(model, ema, loader, optimizer, scaler, loss_fn, use_coarse_loss=True):
    model.train()
    optimizer.zero_grad()
    running_loss = correct = total = 0

    for imgs, fine_lbl, coarse_lbl in loader:
        imgs       = imgs.to(device, non_blocking=True)
        fine_lbl   = fine_lbl.to(device, non_blocking=True)
        coarse_lbl = coarse_lbl.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            s_out, c_out = model(imgs)
            loss = loss_fn(s_out, fine_lbl)
            if use_coarse_loss:
                loss = loss + nn.functional.cross_entropy(c_out, coarse_lbl)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if ema is not None and is_main_process:
            ema.update(model.module if hasattr(model, 'module') else model)

        running_loss += loss.item()
        correct      += (s_out.argmax(1) == fine_lbl).sum().item()
        total        += fine_lbl.size(0)

    if world_size > 1:
        loss_t = torch.tensor(running_loss, device=device)
        corr_t = torch.tensor(correct, device=device)
        tot_t  = torch.tensor(total, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(corr_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(tot_t,  op=dist.ReduceOp.SUM)
        return loss_t.item() / (len(loader) * world_size), 100.0 * corr_t.item() / tot_t.item()

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
            loss       = taxonomy_aware_loss(out, fine_lbl) + nn.functional.cross_entropy(c_out, coarse_lbl)
            running_loss += loss.item()
            correct      += (out.argmax(1) == fine_lbl).sum().item()
            total        += fine_lbl.size(0)

    if world_size > 1:
        loss_t = torch.tensor(running_loss, device=device)
        corr_t = torch.tensor(correct, device=device)
        tot_t  = torch.tensor(total, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(corr_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(tot_t,  op=dist.ReduceOp.SUM)
        return loss_t.item() / (len(loader) * world_size), 100.0 * corr_t.item() / tot_t.item()

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

    print_main(f"\n{'='*52}\n  {label}\n{'='*52}")
    print_main(f"  Accuracy        : {acc:.2f}%")
    print_main(f"  Macro F1        : {macro_f1:.2f}%")
    print_main(f"  Weighted F1     : {weighted_f1:.2f}%")
    print_main(f"  Top-3 Accuracy  : {top3:.2f}%")
    print_main(f"  Top-5 Accuracy  : {top5:.2f}%")
    print_main(f"  ROC-AUC (OvR)   : {roc_auc:.4f}")
    return dict(acc=acc, macro_f1=macro_f1, weighted_f1=weighted_f1, top3=top3, top5=top5, roc_auc=roc_auc)

def full_train_run(backbone_name, loss_fn, use_coarse_loss, use_ema, model_save_path, label=""):
    print_main(f"\n{'#'*60}\n  TRAINING: {label}\n  backbone={backbone_name}, DDP World Size={world_size}\n{'#'*60}")

    base_model = BioHMSC(backbone_name).to(device)
    if world_size > 1:
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = base_model

    ema = ModelEmaV2(base_model, decay=EMA_DECAY) if (use_ema and is_main_process) else None

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
    scaler    = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []
    best_val_acc, best_epoch, early_stop_ctr = 0.0, 0, 0
    full_save_path = f"/kaggle/working/{model_save_path}"
    run_start = time.time()

    for epoch in range(EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        ep_start = time.time()
        t_loss, t_acc = train_one_epoch(model, ema, train_loader, optimizer, scaler, loss_fn, use_coarse_loss)

        if is_main_process:
            eval_model = ema.module if use_ema else (model.module if hasattr(model, 'module') else model)
            v_loss, v_acc = validate(eval_model, val_loader)
            scheduler.step()

            train_loss_hist.append(t_loss)
            val_loss_hist.append(v_loss)
            train_acc_hist.append(t_acc)
            val_acc_hist.append(v_acc)

            elapsed = time.time() - ep_start
            print(f"  Ep {epoch+1:02d}/{EPOCHS} [{elapsed:.0f}s] train_acc={t_acc:.2f}% | val_acc={v_acc:.2f}% | val_loss={v_loss:.4f}")

            if v_acc > best_val_acc:
                best_val_acc, best_epoch, early_stop_ctr = v_acc, epoch + 1, 0
                model_to_save = ema.module if use_ema else (model.module if hasattr(model, 'module') else model)
                torch.save(model_to_save.state_dict(), full_save_path)
                print(f"    ✓ Best model saved ({v_acc:.2f}%)")
            else:
                early_stop_ctr += 1
                if early_stop_ctr >= PATIENCE:
                    print(f"  Early stopping triggered at epoch {epoch+1}")
                    break

    if is_main_process:
        eval_model = BioHMSC(backbone_name).to(device)
        eval_model.load_state_dict(torch.load(full_save_path, weights_only=True))
        return eval_model, {
            "train_loss": train_loss_hist, "val_loss": val_loss_hist,
            "train_acc": train_acc_hist, "val_acc": val_acc_hist,
            "best_val_acc": best_val_acc, "total_mins": (time.time() - run_start) / 60.0
        }
    return None, None

# ============================================================
# 8. COMPLETE MULTI-STAGE RESEARCH PIPELINE
# ============================================================

if __name__ == "__main__":
    # ---------------------------------------------------------
    # STAGE 1: PROPOSED Bio-HMSC+++
    # ---------------------------------------------------------
    print_main("\n" + "="*60 + "\nSTAGE 1: PROPOSED MODEL — Bio-HMSC+++ (DDP Dual T4)\n" + "="*60)
    proposed_model, proposed_hist = full_train_run(
        backbone_name   = "tf_efficientnetv2_m",
        loss_fn         = taxonomy_aware_loss,
        use_coarse_loss = True,
        use_ema         = True,
        model_save_path = "proposed_best.pth",
        label           = "Bio-HMSC+++ (EfficientNetV2-M + TaxLoss + EMA)",
    )

    metrics_proposed = None
    if is_main_process and proposed_model is not None:
        preds_prop, labels_prop, probs_prop = evaluate_full(proposed_model, test_loader, apply_tta=True)
        metrics_proposed = compute_metrics(preds_prop, labels_prop, probs_prop, label="Bio-HMSC+++ (Final Test)")

    # ---------------------------------------------------------
    # STAGE 2: 4-CONFIGURATION ABLATION STUDY
    # ---------------------------------------------------------
    print_main("\n" + "="*60 + "\nSTAGE 2: PROPER 4-CONFIGURATION ABLATION STUDY\n" + "="*60)
    
    # Config A: Standard CE
    model_A, hist_A = full_train_run(
        backbone_name   = "tf_efficientnetv2_m",
        loss_fn         = standard_ce_loss,
        use_coarse_loss = False,
        use_ema         = False,
        model_save_path = "ablation_A.pth",
        label           = "Config A: EfficientNetV2-M + Standard CE",
    )
    metrics_A = None
    if is_main_process and model_A is not None:
        p_A, l_A, pr_A = evaluate_full(model_A, test_loader, apply_tta=False)
        metrics_A = compute_metrics(p_A, l_A, pr_A, label="Config A")

    # Config B: + Taxonomy Loss
    model_B, hist_B = full_train_run(
        backbone_name   = "tf_efficientnetv2_m",
        loss_fn         = taxonomy_aware_loss,
        use_coarse_loss = True,
        use_ema         = False,
        model_save_path = "ablation_B.pth",
        label           = "Config B: + Taxonomy-Aware Loss",
    )
    metrics_B = None
    if is_main_process and model_B is not None:
        p_B, l_B, pr_B = evaluate_full(model_B, test_loader, apply_tta=False)
        metrics_B = compute_metrics(p_B, l_B, pr_B, label="Config B")

    # Config C: + EMA
    model_C, hist_C = full_train_run(
        backbone_name   = "tf_efficientnetv2_m",
        loss_fn         = taxonomy_aware_loss,
        use_coarse_loss = True,
        use_ema         = True,
        model_save_path = "ablation_C.pth",
        label           = "Config C: + EMA Weight Smoothing",
    )
    metrics_C = None
    if is_main_process and model_C is not None:
        p_C, l_C, pr_C = evaluate_full(model_C, test_loader, apply_tta=False)
        metrics_C = compute_metrics(p_C, l_C, pr_C, label="Config C")

    # ---------------------------------------------------------
    # STAGE 3: BASELINE COMPARISONS
    # ---------------------------------------------------------
    print_main("\n" + "="*60 + "\nSTAGE 3: BASELINE COMPARISONS\n" + "="*60)
    BASELINES = [("mobilenetv2_100", "MobileNetV2"), ("efficientnet_b0", "EfficientNet-B0"), ("resnet50", "ResNet50")]
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
        if is_main_process and b_model is not None:
            bp, bl, bpr = evaluate_full(b_model, test_loader, apply_tta=True)
            bm = compute_metrics(bp, bl, bpr, label=f"{b_label} + TTA")
            baseline_results[b_label] = {"model": b_model, "metrics": bm, "hist": b_hist}

    # ---------------------------------------------------------
    # STAGE 4: PAPER VISUALIZATIONS & METRICS (RANK 0 ONLY)
    # ---------------------------------------------------------
    if is_main_process:
        print_main("\n" + "*"*60 + "\nSTAGE 4: EXPORTING PUBLICATION VISUALIZATIONS & TABLES\n" + "*"*60)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(labels_prop, preds_prop)
        plt.figure(figsize=(15, 13))
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=fine_classes, yticklabels=fine_classes)
        plt.title(f"Confusion Matrix — Bio-HMSC+++ (Test Accuracy: {metrics_proposed['acc']:.2f}%)")
        plt.xlabel("Predicted Taxonomic Class")
        plt.ylabel("True Ground Truth Class")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig("/kaggle/working/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Dual Training Curves
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        axes[0].plot(proposed_hist["train_loss"], label="Train Loss", color="blue", lw=2)
        axes[0].plot(proposed_hist["val_loss"], label="Val Loss", color="red", lw=2)
        axes[0].set_title("Training & Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(proposed_hist["train_acc"], label="Train Accuracy", color="blue", lw=2)
        axes[1].plot(proposed_hist["val_acc"], label="Val Accuracy", color="red", lw=2)
        axes[1].set_title("Training & Validation Accuracy (%)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        plt.savefig("/kaggle/working/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. ROC Curves (All 23 classes)
        try:
            y_bin = label_binarize(labels_prop, classes=range(num_classes))
            plt.figure(figsize=(13, 9))
            colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], probs_prop[:, i])
                plt.plot(fpr, tpr, label=fine_classes[i], color=colors[i], lw=1.2)
            plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random (AUC=0.50)")
            plt.title(f"ROC Curves — Bio-HMSC+++ (Macro ROC-AUC: {metrics_proposed['roc_auc']:.4f})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig("/kaggle/working/roc_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"ROC Curve export note: {e}")

        # 4. Print Empirical Summary Tables
        print("\n--- ABLATION STUDY SUMMARY ---")
        if metrics_A and metrics_B and metrics_C:
            print(f"  Config A (Standard CE)            : {metrics_A['acc']:.2f}%")
            print(f"  Config B (+ Taxonomy Loss)        : {metrics_B['acc']:.2f}% [Δ: {metrics_B['acc']-metrics_A['acc']:+.2f}pp]")
            print(f"  Config C (+ EMA)                  : {metrics_C['acc']:.2f}% [Δ: {metrics_C['acc']-metrics_B['acc']:+.2f}pp]")
            print(f"  Config D (Full Bio-HMSC+++ + TTA) : {metrics_proposed['acc']:.2f}% [Δ: {metrics_proposed['acc']-metrics_C['acc']:+.2f}pp]")

        print("\n--- BASELINE COMPARISON TABLE ---")
        for b_label, b_data in baseline_results.items():
            bm = b_data["metrics"]
            pm = b_data["model"]
            params = sum(p.numel() for p in pm.parameters()) / 1e6
            print(f"  {b_label:<22} | Acc: {bm['acc']:>6.2f}% | F1: {bm['macro_f1']:>6.2f}% | AUC: {bm['roc_auc']:.4f} | {params:.1f}M params")
        prop_p = sum(p.numel() for p in proposed_model.parameters()) / 1e6
        print(f"  {'Bio-HMSC+++ (Proposed)':<22} | Acc: {metrics_proposed['acc']:>6.2f}% | F1: {metrics_proposed['macro_f1']:>6.2f}% | AUC: {metrics_proposed['roc_auc']:.4f} | {prop_p:.1f}M params")

    if world_size > 1:
        dist.destroy_process_group()
