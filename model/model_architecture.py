# -*- coding: utf-8 -*-
"""
Bio-HMSC+++ Architecture & Taxonomy Loss Module
================================================
Defines the Hierarchical Multi-Scale Marine Animal Classifier (Bio-HMSC+++)
and the Taxonomy-Aware Loss function with biological phylogenetic penalty matrix.
"""

import torch
import torch.nn as nn
import timm

# ============================================================
# 1. 23 FINE-GRAINED SPECIES & 5 COARSE TAXONOMIC CLASSES
# ============================================================

FINE_CLASSES = [
    "Clams", "Corals", "Crabs", "Dolphin", "Eel", "Fish",
    "Jelly Fish", "Lobster", "Nudibranchs", "Octopus", "Otter",
    "Penguin", "Puffers", "Sea Otter", "Sea Rays", "Sea Urchins",
    "Seahorse", "Seal", "Sharks", "Shrimp", "Squid", "Starfish",
    "Turtle Tortoise"
]

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

COARSE_CLASSES = sorted(list(set(COARSE_MAP.values())))
COARSE_TO_IDX = {c: i for i, c in enumerate(COARSE_CLASSES)}

# ============================================================
# 2. TAXONOMY PENALTY MATRIX BUILDER
# ============================================================

def build_penalty_matrix(classes=FINE_CLASSES, device="cpu"):
    """
    Constructs a phylogenetic distance matrix:
    - Same coarse group (e.g. dolphin vs whale): penalty = 0.5
    - Different coarse group (e.g. dolphin vs jellyfish): penalty = 1.5
    """
    num_c = len(classes)
    matrix = torch.ones(num_c, num_c, device=device)
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            tax_i = COARSE_MAP.get(ci.lower().replace('_', ' '), "invertebrate")
            tax_j = COARSE_MAP.get(cj.lower().replace('_', ' '), "invertebrate")
            if tax_i == tax_j:
                matrix[i, j] = 0.5
            else:
                matrix[i, j] = 1.5
    return matrix

# ============================================================
# 3. BIO-HMSC MODEL ARCHITECTURE
# ============================================================

class BioHMSC(nn.Module):
    """
    Bio-HMSC+++: Biologically-Informed Hierarchical Multi-Scale Classifier
    Backbone: EfficientNetV2-M with Dual Classification Heads
    """
    def __init__(self, 
                 backbone_name: str = "tf_efficientnetv2_m", 
                 num_fine_classes: int = len(FINE_CLASSES), 
                 num_coarse_classes: int = len(COARSE_CLASSES),
                 drop_rate: float = 0.5,
                 drop_path_rate: float = 0.2):
        super().__init__()
        
        # SOTA Feature Extractor
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )
        feat_dim = self.backbone.num_features  # 1280 for EfficientNetV2-M
        
        # Shared MLP Bottleneck
        self.shared_mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(drop_rate),
        )
        
        # Dual Multi-Task Prediction Heads
        self.species_head = nn.Linear(512, num_fine_classes)
        self.coarse_head  = nn.Linear(512, num_coarse_classes)

    def forward(self, x):
        feats = self.backbone(x)
        shared_feats = self.shared_mlp(feats)
        species_logits = self.species_head(shared_feats)
        coarse_logits  = self.coarse_head(shared_feats)
        return species_logits, coarse_logits

# ============================================================
# 4. TAXONOMY-AWARE LOSS
# ============================================================

class TaxonomyAwareLoss(nn.Module):
    """
    Combines Cross-Entropy with Label Smoothing and a Phylogenetic Penalty Term.
    L_total = CE(p, y) + lambda * sum(p_i * penalty_matrix(y, i))
    """
    def __init__(self, penalty_matrix, tax_lambda: float = 0.3, label_smoothing: float = 0.1):
        super().__init__()
        self.penalty_matrix = penalty_matrix
        self.tax_lambda = tax_lambda
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        probs = torch.softmax(logits, dim=1)
        penalties = self.penalty_matrix[targets]
        penalty_term = torch.sum(probs * penalties, dim=1).mean()
        return ce_loss + self.tax_lambda * penalty_term
