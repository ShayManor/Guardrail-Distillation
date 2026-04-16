"""Dataset loading for segmentation tasks."""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2


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
        img = cv2.cvtColor(cv2.imread(str(self.images[idx])), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        lbl = Image.fromarray(cv2.imread(str(self.labels[idx]), cv2.IMREAD_UNCHANGED))

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
            val_size = (self.crop_size, self.crop_size * 2)  # preserve 1:2 aspect ratio
            img = TF.resize(img, val_size, interpolation=TF.InterpolationMode.BILINEAR)
            lbl = TF.resize(lbl, val_size, interpolation=TF.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = self.normalize(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl = self._map_labels(lbl)
        return img, lbl


class IDDDataset(Dataset):
    """India Driving Dataset (IDD Segmentation) loader.

    Expects the tree produced by slurm/data/prep_idd_local.sbatch:
        <root>/leftImg8bit/{train,val}/<drive_id>/*_leftImg8bit.png
        <root>/gtFine/{train,val}/<drive_id>/*_gtFine_labelcsTrainIds.png

    The ``*_csTrainIds.png`` labels come from the AutoNUE toolkit and are
    already mapped to the Cityscapes 19-class trainId ontology (values in
    {0..18, 255}), so this class applies an **identity** label map — unlike
    ``CityscapesDataset`` which applies the 34→19 remap at load time.
    """

    def __init__(self, root, split="val", crop_size=512):
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size

        img_dir = self.root / "leftImg8bit" / split
        lbl_dir = self.root / "gtFine" / split

        self.images = sorted(img_dir.rglob("*_leftImg8bit.png"))
        # Pair each image with its matching cs19 label by stem swap.
        self.labels = []
        for img in self.images:
            rel = img.relative_to(img_dir)
            lbl = lbl_dir / rel.parent / img.name.replace(
                "_leftImg8bit.png", "_gtFine_labelcsTrainIds.png"
            )
            self.labels.append(lbl)

        missing = [p for p in self.labels if not p.exists()]
        assert not missing, (
            f"IDDDataset: {len(missing)} labels missing (e.g. {missing[:3]}). "
            "Did the AutoNUE toolkit remap finish?"
        )
        assert len(self.images) == len(self.labels) and self.images, (
            f"IDDDataset: no images under {img_dir}"
        )

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(str(self.images[idx])), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        lbl = Image.fromarray(cv2.imread(str(self.labels[idx]), cv2.IMREAD_UNCHANGED))

        # IDD is eval-only for us — use the same val resize as CityscapesDataset
        # so the student input distribution matches what it saw during training.
        val_size = (self.crop_size, self.crop_size * 2)
        img = TF.resize(img, val_size, interpolation=TF.InterpolationMode.BILINEAR)
        lbl = TF.resize(lbl, val_size, interpolation=TF.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = self.normalize(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        # Identity label map: values are already cs19 trainIds in {0..18, 255}.
        return img, lbl


class BDDDataset(Dataset):
    """BDD100K Segmentation loader (Kaggle solesensei mirror).

    Expects the tree produced by slurm/data/prep_bdd.sbatch:
        <root>/seg/images/{train,val}/*.jpg
        <root>/seg/labels/{train,val}/*_train_id.png

    Labels ship as Cityscapes 19-class trainIds (values in {0..18, 255}),
    so this class applies an identity label map — same pattern as IDDDataset.
    """

    def __init__(self, root, split="val", crop_size=512):
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size

        img_dir = self.root / "seg" / "images" / split
        lbl_dir = self.root / "seg" / "labels" / split

        self.images = sorted(img_dir.glob("*.jpg"))
        self.labels = [lbl_dir / f"{p.stem}_train_id.png" for p in self.images]

        missing = [p for p in self.labels if not p.exists()]
        assert not missing, (
            f"BDDDataset: {len(missing)} labels missing (e.g. {missing[:3]}). "
            "Did prep_bdd.sbatch finish?"
        )
        assert len(self.images) == len(self.labels) and self.images, (
            f"BDDDataset: no images under {img_dir}"
        )

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(str(self.images[idx])), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        lbl = Image.fromarray(cv2.imread(str(self.labels[idx]), cv2.IMREAD_UNCHANGED))

        val_size = (self.crop_size, self.crop_size * 2)
        img = TF.resize(img, val_size, interpolation=TF.InterpolationMode.BILINEAR)
        lbl = TF.resize(lbl, val_size, interpolation=TF.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = self.normalize(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
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

        val_size = (self.crop_size, self.crop_size * 2)  # preserve 1:2 aspect ratio
        img = TF.resize(img, val_size, interpolation=TF.InterpolationMode.BILINEAR)
        lbl = TF.resize(lbl, val_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.split == "train" and torch.rand(1) > 0.5:
            img = TF.hflip(img)
            lbl = TF.hflip(lbl)

        img = TF.to_tensor(img)
        img = self.normalize(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl


def build_dataloaders(cfg):
    """Build train/val dataloaders from config.

    Uses ``cfg.seed`` to pin:
      - the train shuffle permutation (via a torch.Generator)
      - each worker's python/numpy/torch RNG state (via seed_worker)
    The val loader is unshuffled and does not need a generator.
    """
    from src.train.utils import seed_worker

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

    train_gen = torch.Generator()
    train_gen.manual_seed(int(cfg.seed))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        generator=train_gen,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader