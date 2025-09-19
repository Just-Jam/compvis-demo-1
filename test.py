from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import streamlit as st
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd
import re
from torchvision.models import resnet18
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import random

# Use these when you create DataLoaders later
DATALOADER_KW = dict(
    num_workers=0,          # <— main fix: avoid background workers in notebooks
    pin_memory=False,       # safe default; you can set True if using CUDA
    persistent_workers=False
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", DEVICE)

# Paths
TRAIN_DIR = Path("data/train")
VAL_DIR   = Path("data/validation")

TRAIN_CSV = TRAIN_DIR / "labels.csv"
VAL_CSV   = VAL_DIR / "labels.csv"

# Accept either correct or misspelled folder names
SUBDIR_ALIASES = {
    "pedestrian":      ["pedestrian", "pedestrain"],
    "no pedestrian":   ["no pedestrian", "no pedestrain"],
}

# For display & mapping
CLASS_NAMES = ["no pedestrian", "pedestrian"]   # index 0 -> no pedestrian, 1 -> pedestrian
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# --- 2) Dataset from CSV (normalizes labels, finds files even if folder names differ)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def normalize_label(s: str) -> str:
    s = s.strip().lower()
    # unify spellings
    s = s.replace("pedestrain", "pedestrian")
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    # ensure "no pedestrian" has the leading 'no'
    if s in {"no pedestrian", "no pedestrain"}:
        s = "no pedestrian"
    if s == "pedestrian":
        return "pedestrian"
    if s == "no pedestrian":
        return "no pedestrian"
    raise ValueError(f"Unknown label in CSV: {s}")

def find_image_path(base_dir: Path, normalized_label: str, filename: str) -> Path:
    # Try each alias directory until we find the file
    for sub in SUBDIR_ALIASES[normalized_label]:
        candidate = base_dir / sub / filename
        if candidate.exists():
            return candidate
    # If not found, also try recursive search under aliased dirs (costly, but safe)
    for sub in SUBDIR_ALIASES[normalized_label]:
        root = base_dir / sub
        if root.exists():
            hits = list(root.rglob(filename))
            if hits:
                return hits[0]
    # Lastly, if CSV has relative paths (e.g., "pedestrian/img001.jpg"), try them directly
    rel_try = base_dir / filename
    if rel_try.exists():
        return rel_try
    raise FileNotFoundError(f"Could not locate image '{filename}' for label '{normalized_label}' under {base_dir}")

class CsvImageDataset(Dataset):
    def __init__(self, base_dir: Path, csv_path: Path, transform=None):
        assert base_dir.exists(), f"Base dir not found: {base_dir}"
        assert csv_path.exists(), f"CSV not found: {csv_path}"
        self.base_dir = base_dir
        self.df = pd.read_csv(csv_path)
        # Expecting columns: image, label
        assert {"image", "label"}.issubset(self.df.columns), "CSV must have columns: image, label"
        self.transform = transform

        # Pre-resolve absolute paths & targets
        self.samples = []
        for _, row in self.df.iterrows():
            raw_label = str(row["label"])
            norm_label = normalize_label(raw_label)
            fname = str(row["image"])
            # Allow CSV to hold either bare filename (img.jpg) or relative path (pedestrian/img.jpg)
            # If it's a relative path with subdir, prefer that; else use aliases.
            if "/" in fname or "\\" in fname:
                p = (self.base_dir / fname).resolve()
                if not p.exists():
                    # try to strip leading subdir and resolve via aliases
                    fname = Path(fname).name
                    p = find_image_path(self.base_dir, norm_label, fname)
            else:
                p = find_image_path(self.base_dir, norm_label, fname)

            target = CLASS_TO_IDX[norm_label]
            self.samples.append((p, target))

        print(f"Loaded {len(self.samples)} samples from {csv_path.name}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, target

def denormalize(img_tensor):
    """Undo ImageNet normalization to display images correctly."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean  # unnormalize
    img = np.clip(img, 0, 1)
    return img

def show_class_examples(model, val_loader, class_names, samples_per_class=4):
    model.eval()
    collected = {cls: [] for cls in range(len(class_names))}

    # Flatten the whole loader into one list of (image, label, pred)
    all_examples = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for i in range(len(labels)):
                all_examples.append((images[i], labels[i], preds[i]))

    # Shuffle so you get different picks each run
    random.shuffle(all_examples)

    # Collect up to samples_per_class for each class
    for img, label, pred in all_examples:
        cls = label.item()
        if len(collected[cls]) < samples_per_class:
            collected[cls].append((img, label, pred))
        if all(len(v) >= samples_per_class for v in collected.values()):
            break

    # --- Plot
    total = samples_per_class * len(class_names)
    plt.figure(figsize=(15, 6))
    idx = 1
    for cls, examples in collected.items():
        for (img, label, pred) in examples:
            plt.subplot(len(class_names), samples_per_class, idx)
            idx += 1
            img = denormalize(img)
            plt.imshow(img)
            plt.title(
                f"T: {class_names[label.item()]}\nP: {class_names[pred.item()]}",
                color="green" if pred == label else "red",
                fontsize=9
            )
            plt.axis("off")
    plt.tight_layout()
    plt.show()

checkpoint = torch.load(
    "models/resnet18_pedestrian.pt",
    map_location=DEVICE  # load to CPU first, we'll move to GPU if available
)

# Recreate same model architecture
model = resnet18(weights=None)   # no pretrained weights here, we only load your trained weights
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model.load_state_dict(checkpoint["model"])
model = model.to(DEVICE)

# 2. Reuse the class names (stored in checkpoint or from your earlier variable)
CLASS_NAMES = checkpoint.get("class_names", ["no pedestrian", "pedestrian"])

# Load validation data
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])
val_ds   = CsvImageDataset(VAL_DIR,   VAL_CSV,   transform=val_tfms)
BATCH_SIZE = 32
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **DATALOADER_KW)

# 3. Show predictions
show_class_examples(model, val_loader, CLASS_NAMES, samples_per_class=4)