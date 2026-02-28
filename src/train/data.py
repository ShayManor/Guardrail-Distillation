"""Dataset loading for segmentation tasks."""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


# ── Cityscapes label mapping (34 → 19 train IDs) ──
CITYSCAPES_LABEL_MAP = {
    -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255,
    15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18,
}


class CityscapesDataset(Dataset):
    """Standard Cityscapes from local directory."""

    def __init__(self, root, split="train", crop_size=512):
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size

        img_dir = self.root / "leftImg8bit" / split
        lbl_dir = self.root / "gtFine" / split

        self.images = sorted(img_dir.rglob("*_leftImg8bit.png"))
        self.labels = sorted(lbl_dir.rglob("*_gtFine_labelIds.png"))
        assert len(self.images) == len(self.labels), (
            f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels"
        )

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def _map_labels(self, lbl):
        mapped = torch.full_like(lbl, 255)
        for k, v in CITYSCAPES_LABEL_MAP.items():
            mapped[lbl == k] = v
        return mapped

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        lbl = Image.open(self.labels[idx])

        # Joint random crop + flip for training
        if self.split == "train":
            i, j, h, w = T.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
            img = TF.crop(img, i, j, h, w)
            lbl = TF.crop(lbl, i, j, h, w)
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)
        else:
            # Resize for val
            img = TF.resize(img, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.BILINEAR)
            lbl = TF.resize(lbl, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = self.normalize(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl = self._map_labels(lbl)
        return img, lbl


class HFSegmentationDataset(Dataset):
    """Load a segmentation dataset from HuggingFace."""

    def __init__(self, dataset, crop_size=512, split="train",
                 image_key="image", label_key="semantic_segmentation"):
        self.ds = dataset
        self.crop_size = crop_size
        self.split = split
        self.image_key = image_key
        self.label_key = label_key
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample[self.image_key].convert("RGB")
        lbl = sample[self.label_key]

        img = TF.resize(img, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.BILINEAR)
        lbl = TF.resize(lbl, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.NEAREST)

        if self.split == "train" and torch.rand(1) > 0.5:
            img = TF.hflip(img)
            lbl = TF.hflip(lbl)

        img = TF.to_tensor(img)
        img = self.normalize(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl


def build_dataloaders(cfg):
    """Build train/val dataloaders from config."""
    path = cfg.dataset_path

    if path.startswith("hf://"):
        from datasets import load_dataset
        hf_name = path.replace("hf://", "")
        raw = load_dataset(hf_name)
        train_ds = HFSegmentationDataset(raw["train"], cfg.crop_size, "train")
        val_ds = HFSegmentationDataset(raw["validation"], cfg.crop_size, "val")
    else:
        train_ds = CityscapesDataset(path, "train", cfg.crop_size)
        val_ds = CityscapesDataset(path, "val", cfg.crop_size)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    return train_loader, val_loader