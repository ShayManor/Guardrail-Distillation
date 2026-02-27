"""Dataset loading — local directories and HuggingFace streaming."""
import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image
from torchvision import transforms as T

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
    ds = load_dataset(dataset_id, split=split, streaming=True, trust_remote_code=True)

    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        img = sample[image_key]
        lbl = sample[label_key]
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
        raise FileNotFoundError(f"No images in {root}. Expected '{images_subdir}/' or 'leftImg8bit/'.")

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    img_files = sorted(f for f in img_dir.rglob("*") if f.suffix.lower() in exts)

    for i, img_path in enumerate(img_files):
        if max_samples and i >= max_samples:
            break

        rel = img_path.relative_to(img_dir)
        lbl_path = None
        for lbl_ext in exts:
            candidate = lbl_dir / rel.with_suffix(lbl_ext)
            cs_name = rel.stem.replace("_leftImg8bit", "_gtFine_labelIds")
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


CITYSCAPES_PALETTE = None  # lazy-loaded

def _color_label_to_ids(color_img: Image.Image) -> Image.Image:
    """Convert Cityscapes color-coded label image to label ID image."""
    # Cityscapes color -> trainId mapping (19 classes)
    color_to_id = {
        (128, 64, 128): 0,   # road
        (244, 35, 232): 1,   # sidewalk
        (70, 70, 70): 2,     # building
        (102, 102, 156): 3,  # wall
        (190, 153, 153): 4,  # fence
        (153, 153, 153): 5,  # pole
        (250, 170, 30): 6,   # traffic light
        (220, 220, 0): 7,    # traffic sign
        (107, 142, 35): 8,   # vegetation
        (152, 251, 152): 9,  # terrain
        (70, 130, 180): 10,  # sky
        (220, 20, 60): 11,   # person
        (255, 0, 0): 12,     # rider
        (0, 0, 142): 13,     # car
        (0, 0, 70): 14,      # truck
        (0, 60, 100): 15,    # bus
        (0, 80, 100): 16,    # train
        (0, 0, 230): 17,     # motorcycle
        (119, 11, 32): 18,   # bicycle
    }
    arr = np.array(color_img.convert("RGB"))
    out = np.full(arr.shape[:2], 255, dtype=np.uint8)
    for color, idx in color_to_id.items():
        mask = np.all(arr == color, axis=-1)
        out[mask] = idx
    return Image.fromarray(out)


def load_cityscapes_paired(
    images_dir: Path,
    max_samples: Optional[int] = None,
) -> Iterator[tuple[str, Image.Image, Image.Image]]:
    """Load side-by-side paired Cityscapes images (left=photo, right=color label)."""
    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(f for f in images_dir.rglob("*") if f.suffix.lower() in exts)
    for i, p in enumerate(files):
        if max_samples and i >= max_samples:
            break
        img = Image.open(p).convert("RGB")
        w, h = img.size
        half = w // 2
        photo = img.crop((0, 0, half, h))
        color_lbl = img.crop((half, 0, w, h))
        label = _color_label_to_ids(color_lbl)
        yield p.stem, photo, label

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

    if not dest.exists():
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
    root = _find_deepest_images_dir(root)
    yield from load_cityscapes_paired(root, max_samples)
