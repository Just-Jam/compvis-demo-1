# --- 2) Dataset from CSV (normalizes labels, finds files even if folder names differ)
import os, re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
# Accept either correct or misspelled folder names
SUBDIR_ALIASES = {
    "pedestrian":      ["pedestrian", "pedestrain"],
    "no pedestrian":   ["no pedestrian", "no pedestrain"],
}

CLASS_NAMES = ["no pedestrian", "pedestrian"]   # index 0 -> no pedestrian, 1 -> pedestrian
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

def denormalize(img_tensor):
    """Undo ImageNet normalization to display images correctly."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean  # unnormalize
    img = np.clip(img, 0, 1)
    return img


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