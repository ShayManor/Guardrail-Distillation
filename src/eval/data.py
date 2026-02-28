"""Dataset loading — local directories and HuggingFace streaming."""
import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image
from torchvision import transforms as T

CITYSCAPES_LABELID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
    14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255,
}

CITYSCAPES_COLOR_TO_TRAINID = {
    (128, 64, 128): 0,   (244, 35, 232): 1,   (70, 70, 70): 2,
    (102, 102, 156): 3,  (190, 153, 153): 4,  (153, 153, 153): 5,
    (250, 170, 30): 6,   (220, 220, 0): 7,    (107, 142, 35): 8,
    (152, 251, 152): 9,  (70, 130, 180): 10,  (220, 20, 60): 11,
    (255, 0, 0): 12,     (0, 0, 142): 13,     (0, 0, 70): 14,
    (0, 60, 100): 15,    (0, 80, 100): 16,    (0, 0, 230): 17,
    (119, 11, 32): 18,
}

DEFAULT_TRANSFORM = T.Compose([
    T.Resize((512, 1024)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LABEL_TRANSFORM = T.Compose([
    T.Resize((512, 1024), interpolation=T.InterpolationMode.NEAREST),
])

IGNORE_INDEX = 255


def is_hf_path(path: str) -> bool:
    return path.startswith("hf://") or path.startswith("huggingface://")


def is_kaggle_path(path: str) -> bool:
    return path.startswith("kaggle://")


def _parse_hf_path(path: str) -> tuple[str, Optional[str]]:
    cleaned = path.replace("hf://", "").replace("huggingface://", "").strip("/")
    parts = cleaned.split("/")
    if len(parts) >= 3:
        return "/".join(parts[:2]), parts[2]
    return "/".join(parts[:2]), None


def load_hf_stream(
    path: str,
    image_key: str = "image",
    label_key: str = "label",
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[tuple[str, Image.Image, Image.Image]]:
    """Stream dataset from HuggingFace without downloading to disk."""
    from datasets import load_dataset

    dataset_id, parsed_split = _parse_hf_path(path)
    split = split or parsed_split or "validation"
    ds = load_dataset(dataset_id, split=split)

    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        img = sample[image_key]
        lbl = sample[label_key]
        arr = np.array(lbl)
        print(f"[DEBUG] lbl type={type(lbl)} arr.shape={arr.shape} arr.ndim={arr.ndim}")
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        if not isinstance(lbl, Image.Image):
            lbl = Image.fromarray(np.array(lbl).astype(np.uint8))
        yield f"hf_{i:06d}", img, lbl


def load_local_dataset(
    path: str,
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    max_samples: Optional[int] = None,
) -> Iterator[tuple[str, Image.Image, Image.Image]]:
    """Load from local dir with matching image/label filenames."""
    root = Path(path)
    img_dir = root / images_subdir
    lbl_dir = root / labels_subdir

    if not img_dir.exists():
        img_dir = root / "leftImg8bit"
        lbl_dir = root / "gtFine"
    if not img_dir.exists():
        img_dir = root / "img"
        lbl_dir = root / "label"
    if not img_dir.exists():
        raise FileNotFoundError(f"No images in {root}. Expected '{images_subdir}/', 'leftImg8bit/', or 'img/'.")

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    img_files = sorted(f for f in img_dir.rglob("*") if f.suffix.lower() in exts)

    for i, img_path in enumerate(img_files):
        if max_samples and i >= max_samples:
            break

        rel = img_path.relative_to(img_dir)
        lbl_path = None
        for lbl_ext in exts:
            candidate = lbl_dir / rel.with_suffix(lbl_ext)
            cs_name = rel.stem.replace("_leftImg8bit", "_gtFine_labelTrainIds")
            candidate2 = lbl_dir / rel.parent / (cs_name + lbl_ext)
            if candidate.exists():
                lbl_path = candidate
                break
            if candidate2.exists():
                lbl_path = candidate2
                break

        if lbl_path is None:
            print(f"[WARN] No label for {img_path.name}, skipping")
            continue

        yield img_path.stem, Image.open(img_path).convert("RGB"), Image.open(lbl_path)

def _find_deepest_images_dir(base: Path) -> Path:
    """Walk down to find the deepest directory containing image files."""
    image_exts = {".jpg", ".jpeg", ".png"}
    best = base
    for dirpath, _, filenames in os.walk(base):
        if any(Path(f).suffix.lower() in image_exts for f in filenames):
            best = Path(dirpath)
    return best


def _parse_kaggle_path(path: str) -> tuple[str, Optional[str]]:
    """'kaggle://owner/dataset[/split]' -> (owner/dataset, split|None)"""
    cleaned = path.replace("kaggle://", "").strip("/")
    parts = cleaned.split("/")
    if len(parts) >= 3:
        return "/".join(parts[:2]), parts[2]
    return "/".join(parts[:2]), None


def load_kaggle_dataset(
    path: str,
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[tuple[str, Image.Image, Image.Image]]:
    """
    Download a Kaggle dataset (cached) and iterate over image/label pairs.
    Requires kaggle credentials (~/.kaggle/kaggle.json or env vars).

    Path format: kaggle://owner/dataset-name[/split]
    Downloads once to cache_dir (default: ~/.cache/kaggle_datasets/).
    """
    import subprocess
    import zipfile

    dataset_id, split = _parse_kaggle_path(path)
    safe_name = dataset_id.replace("/", "_")
    cache = Path(cache_dir or Path.home() / ".cache" / "kaggle_datasets")
    dest = cache / safe_name

    if not dest.exists() or not any(dest.iterdir()):
        print(f"[kaggle] Downloading {dataset_id} -> {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        import shutil
        kaggle_bin = shutil.which("kaggle") or "/workspace/.venv/bin/kaggle"

        result = subprocess.run(
            [kaggle_bin, "datasets", "download", "-d", dataset_id, "-p", str(dest), "--unzip"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Try as competition dataset
            result = subprocess.run(
                [kaggle_bin, "competitions", "download", "-c", dataset_id.split("/")[-1], "-p", str(dest)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Kaggle download failed: {result.stderr}")
            # Unzip any remaining zips
            for zf in dest.glob("*.zip"):
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(dest)
                zf.unlink()

        print(f"[kaggle] Downloaded to {dest}")
    else:
        print(f"[kaggle] Using cached {dest}")

    # If split specified, look for subdir
    root = dest / split if split and (dest / split).exists() else dest

    # Auto-detect format: separate dirs vs side-by-side paired
    has_img_dir = (root / images_subdir).exists() or (root / "leftImg8bit").exists()
    has_gtfine = (root / labels_subdir).exists() or (root / "gtFine").exists()

    yield from load_local_dataset(str(root), images_subdir, labels_subdir, max_samples)
