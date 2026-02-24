"""Dataset loading — local directories and HuggingFace streaming."""

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
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(dest), "--unzip"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Try as competition dataset
            result = subprocess.run(
                ["kaggle", "competitions", "download", "-c", dataset_id.split("/")[-1], "-p", str(dest)],
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

    yield from load_local_dataset(str(root), images_subdir, labels_subdir, max_samples)
