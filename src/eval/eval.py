"""
Guardrail Distillation Evaluation Pipeline
==========================================
Evaluates segmentation models (teacher, student, KD variants) on local or
HuggingFace-streamed datasets. Outputs per-image and aggregate metrics to CSV.
Includes visualization utilities.

Usage:
    from eval_pipeline import run_eval, plot_results

    # HF-streamed dataset
    run_eval(
        model_path="./checkpoints/student_kd",
        dataset_path="zurich-dark/val",          # local
        # OR dataset_path="hf://org/dark_zurich",  # streams from HF
        output_csv="results/kd_darkzurich.csv",
        num_classes=19,
    )

    # Visualize
    plot_results("results/kd_darkzurich.csv", save_dir="results/figures")
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

# Config

CITYSCAPES_PALETTE = [
    128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
    153, 153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152,
    70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100,
    0, 80, 100, 0, 0, 230, 119, 11, 32,
]

DEFAULT_TRANSFORM = T.Compose([
    T.Resize((512, 1024)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LABEL_TRANSFORM = T.Compose([
    T.Resize((512, 1024), interpolation=T.InterpolationMode.NEAREST),
])

IGNORE_INDEX = 255


# Metrics (per-image + aggregate)

@dataclass
class ImageMetrics:
    image_id: str = ""
    miou: float = 0.0
    pixel_acc: float = 0.0
    mean_acc: float = 0.0
    msp_mean: float = 0.0          # mean softmax probability (confidence)
    msp_std: float = 0.0
    entropy_mean: float = 0.0
    entropy_std: float = 0.0
    per_class_iou: str = ""        # JSON-encoded dict
    num_pixels: int = 0
    num_ignored: int = 0
    inference_ms: float = 0.0


def compute_metrics(
    pred_logits: torch.Tensor,   # (1, C, H, W)
    label: torch.Tensor,         # (H, W) with ignore=255
    num_classes: int,
    image_id: str = "",
    inference_ms: float = 0.0,
) -> ImageMetrics:
    """Compute segmentation + uncertainty metrics for a single image."""
    C = num_classes
    probs = F.softmax(pred_logits.squeeze(0), dim=0)          # (C, H, W)
    pred = probs.argmax(dim=0)                                 # (H, W)
    label = label.squeeze()

    valid = label != IGNORE_INDEX
    pred_v = pred[valid]
    label_v = label[valid]
    probs_v = probs[:, valid]

    # Per-class IoU
    per_class_iou = {}
    ious = []
    for c in range(C):
        p_c = pred_v == c
        l_c = label_v == c
        inter = (p_c & l_c).sum().item()
        union = (p_c | l_c).sum().item()
        if union > 0:
            iou = inter / union
            ious.append(iou)
            per_class_iou[c] = round(iou, 5)

    # Pixel accuracy
    correct = (pred_v == label_v).sum().item()
    total = valid.sum().item()
    pixel_acc = correct / max(total, 1)

    # Mean class accuracy
    class_accs = []
    for c in range(C):
        mask = label_v == c
        if mask.sum() > 0:
            class_accs.append((pred_v[mask] == c).float().mean().item())
    mean_acc = np.mean(class_accs) if class_accs else 0.0

    # Confidence (MSP)
    max_probs = probs_v.max(dim=0).values
    msp_mean = max_probs.mean().item()
    msp_std = max_probs.std().item()

    # Entropy
    eps = 1e-8
    ent = -(probs_v * (probs_v + eps).log()).sum(dim=0)
    entropy_mean = ent.mean().item()
    entropy_std = ent.std().item()

    return ImageMetrics(
        image_id=image_id,
        miou=np.mean(ious) if ious else 0.0,
        pixel_acc=pixel_acc,
        mean_acc=mean_acc,
        msp_mean=msp_mean,
        msp_std=msp_std,
        entropy_mean=entropy_mean,
        entropy_std=entropy_std,
        per_class_iou=json.dumps(per_class_iou),
        num_pixels=total,
        num_ignored=int((~valid).sum().item()),
        inference_ms=inference_ms,
    )


# Dataset Loading (local or HF streaming)

def _is_hf_path(path: str) -> bool:
    return path.startswith("hf://") or path.startswith("huggingface://")


def _parse_hf_path(path: str) -> tuple[str, Optional[str]]:
    """Parse 'hf://org/dataset' or 'hf://org/dataset/split' -> (dataset_id, split)."""
    cleaned = path.replace("hf://", "").replace("huggingface://", "")
    parts = cleaned.strip("/").split("/")
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
    """Load from local directory: path/images/, path/labels/ with matching filenames."""
    root = Path(path)
    img_dir = root / images_subdir
    lbl_dir = root / labels_subdir

    if not img_dir.exists():
        # Fallback: try leftImg8bit / gtFine style (Cityscapes)
        img_dir = root / "leftImg8bit"
        lbl_dir = root / "gtFine"

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Cannot find images in {root}. Expected '{images_subdir}/' or 'leftImg8bit/'."
        )

    # Recursively find images
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    img_files = sorted(
        f for f in img_dir.rglob("*") if f.suffix.lower() in exts
    )

    for i, img_path in enumerate(img_files):
        if max_samples and i >= max_samples:
            break

        # Find matching label
        rel = img_path.relative_to(img_dir)
        lbl_path = None
        for lbl_ext in exts:
            candidate = lbl_dir / rel.with_suffix(lbl_ext)
            # Also try common Cityscapes naming
            cs_name = rel.stem.replace("_leftImg8bit", "_gtFine_labelIds")
            candidate2 = lbl_dir / rel.parent / (cs_name + lbl_ext)
            if candidate.exists():
                lbl_path = candidate
                break
            if candidate2.exists():
                lbl_path = candidate2
                break

        if lbl_path is None:
            print(f"[WARN] No label found for {img_path.name}, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path)
        yield img_path.stem, img, lbl


# Model Loading

def load_model(
    model_path: str,
    device: torch.device,
    num_classes: int = 19,
    model_factory: Optional[Callable] = None,
) -> torch.nn.Module:
    """
    Load a segmentation model from a local path.

    Supports:
      - TorchScript (.pt/.ts): torch.jit.load
      - State dict (.pth/.ckpt): requires model_factory to instantiate arch
      - Full checkpoint with 'model_state_dict' or 'state_dict' key

    Args:
        model_path: Path to model file.
        device: Target device.
        num_classes: Number of segmentation classes.
        model_factory: Callable() -> nn.Module, needed for state-dict checkpoints.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    suffix = path.suffix.lower()

    # TorchScript
    if suffix in (".pt", ".ts"):
        try:
            model = torch.jit.load(str(path), map_location=device)
            model.eval()
            return model
        except Exception:
            pass  # Fall through to state_dict loading

    # State dict
    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    if isinstance(ckpt, torch.nn.Module):
        ckpt.eval()
        return ckpt.to(device)

    # Extract state dict from checkpoint wrapper
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                ckpt = ckpt[key]
                break

    if model_factory is None:
        raise ValueError(
            "Checkpoint is a state_dict but no model_factory provided. "
            "Pass model_factory=lambda: YourModel(num_classes=N)."
        )

    model = model_factory()
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model


# Core Eval Loop

@torch.no_grad()
def run_eval(
    model_path: str,
    dataset_path: str,
    output_csv: str,
    num_classes: int = 19,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_factory: Optional[Callable] = None,
    batch_size: int = 1,  # kept for future batched eval
    max_samples: Optional[int] = None,
    image_transform: Optional[Callable] = None,
    label_transform: Optional[Callable] = None,
    # HF-specific
    hf_image_key: str = "image",
    hf_label_key: str = "label",
    hf_split: Optional[str] = None,
    # Local-specific
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Path:
    """
    Run evaluation and write per-image metrics to CSV.

    Returns path to the output CSV.
    """
    dev = torch.device(device)
    img_tf = image_transform or DEFAULT_TRANSFORM
    lbl_tf = label_transform or LABEL_TRANSFORM

    # Load model
    print(f"[eval] Loading model from {model_path}")
    model = load_model(model_path, dev, num_classes, model_factory)

    # Dataset iterator
    if _is_hf_path(dataset_path):
        print(f"[eval] Streaming dataset from HuggingFace: {dataset_path}")
        data_iter = load_hf_stream(
            dataset_path, hf_image_key, hf_label_key, hf_split, max_samples
        )
    else:
        print(f"[eval] Loading local dataset: {dataset_path}")
        data_iter = load_local_dataset(
            dataset_path, images_subdir, labels_subdir, max_samples
        )

    # Prepare output
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[ImageMetrics] = []
    total_time = 0.0
    count = 0

    # Header metadata
    meta = {
        "model_path": model_path,
        "model_name": model_name or Path(model_path).stem,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name or dataset_path,
        "num_classes": num_classes,
        "device": device,
    }

    for img_id, pil_img, pil_lbl in data_iter:
        # Preprocess
        img_tensor = img_tf(pil_img).unsqueeze(0).to(dev)      # (1, 3, H, W)
        lbl_np = np.array(lbl_tf(pil_lbl)).astype(np.int64)
        lbl_tensor = torch.from_numpy(lbl_np).long().to(dev)    # (H, W)

        # Inference
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(img_tensor)

        # Handle various output formats
        if isinstance(out, dict):
            logits = out.get("out") or out.get("logits") or next(iter(out.values()))
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        # Ensure logits match label spatial dims
        if logits.shape[-2:] != lbl_tensor.shape[-2:]:
            logits = F.interpolate(
                logits, size=lbl_tensor.shape[-2:], mode="bilinear", align_corners=False
            )

        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Metrics
        m = compute_metrics(logits, lbl_tensor, num_classes, img_id, elapsed_ms)
        results.append(m)
        total_time += elapsed_ms
        count += 1

        if count % 50 == 0:
            running_miou = np.mean([r.miou for r in results])
            print(f"  [{count}] running mIoU={running_miou:.4f}  last={elapsed_ms:.1f}ms")

    if count == 0:
        print("[eval] WARNING: No images processed.")
        return out_path

    # Write CSV
    fieldnames = list(ImageMetrics.__dataclass_fields__.keys())
    meta_fields = ["model_name", "dataset_name"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_fields + fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["model_name"] = meta["model_name"]
            row["dataset_name"] = meta["dataset_name"]
            writer.writerow(row)

    # Summary
    agg_miou = np.mean([r.miou for r in results])
    agg_pacc = np.mean([r.pixel_acc for r in results])
    agg_msp = np.mean([r.msp_mean for r in results])
    agg_ent = np.mean([r.entropy_mean for r in results])
    avg_ms = total_time / count

    print(f"\n{'='*60}")
    print(f"  Model:      {meta['model_name']}")
    print(f"  Dataset:    {meta['dataset_name']}")
    print(f"  Samples:    {count}")
    print(f"  mIoU:       {agg_miou:.4f}")
    print(f"  Pixel Acc:  {agg_pacc:.4f}")
    print(f"  Mean MSP:   {agg_msp:.4f}")
    print(f"  Mean Ent:   {agg_ent:.4f}")
    print(f"  Avg Inf:    {avg_ms:.1f} ms/img")
    print(f"  Output:     {out_path}")
    print(f"{'='*60}\n")

    return out_path


# Visualization

def plot_results(
    csv_path: str | list[str],
    save_dir: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Generate analysis plots from one or more eval CSVs.

    Plots:
      1. mIoU distribution (histogram + KDE)
      2. Confidence vs mIoU scatter (reliability diagram proxy)
      3. Per-class IoU bar chart
      4. Entropy vs mIoU scatter
      5. Risk-coverage curve (AURC)
      6. Inference time distribution

    If multiple CSVs, overlays for comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if isinstance(csv_path, str):
        csv_path = [csv_path]

    dfs = []
    for p in csv_path:
        df = pd.read_csv(p)
        if "model_name" not in df.columns:
            df["model_name"] = Path(p).stem
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    models = combined["model_name"].unique()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Segmentation Evaluation Results", fontsize=14, fontweight="bold")

    # 1. mIoU distribution
    ax = axes[0, 0]
    for m in models:
        sub = combined[combined["model_name"] == m]
        ax.hist(sub["miou"], bins=30, alpha=0.5, label=m, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("mIoU")
    ax.set_ylabel("Count")
    ax.set_title("mIoU Distribution")
    ax.legend(fontsize=8)

    # 2. Confidence (MSP) vs mIoU
    ax = axes[0, 1]
    for m in models:
        sub = combined[combined["model_name"] == m]
        ax.scatter(sub["msp_mean"], sub["miou"], alpha=0.3, s=10, label=m)
    ax.set_xlabel("Mean Softmax Prob (Confidence)")
    ax.set_ylabel("mIoU")
    ax.set_title("Confidence vs mIoU")
    ax.legend(fontsize=8)

    # 3. Per-class IoU (aggregate)
    ax = axes[0, 2]
    for m in models:
        sub = combined[combined["model_name"] == m]
        all_class_ious: dict[int, list[float]] = {}
        for _, row in sub.iterrows():
            try:
                pci = json.loads(row["per_class_iou"])
            except (json.JSONDecodeError, TypeError):
                continue
            for k, v in pci.items():
                all_class_ious.setdefault(int(k), []).append(v)
        classes = sorted(all_class_ious.keys())
        means = [np.mean(all_class_ious[c]) for c in classes]
        ax.bar(
            [c + 0.2 * list(models).index(m) for c in classes],
            means, width=0.2, alpha=0.7, label=m,
        )
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Mean IoU")
    ax.set_title("Per-Class IoU")
    ax.legend(fontsize=8)

    # 4. Entropy vs mIoU
    ax = axes[1, 0]
    for m in models:
        sub = combined[combined["model_name"] == m]
        ax.scatter(sub["entropy_mean"], sub["miou"], alpha=0.3, s=10, label=m)
    ax.set_xlabel("Mean Entropy")
    ax.set_ylabel("mIoU")
    ax.set_title("Entropy vs mIoU")
    ax.legend(fontsize=8)

    # 5. Risk-Coverage curve (using MSP as selector)
    ax = axes[1, 1]
    for m in models:
        sub = combined[combined["model_name"] == m].sort_values("msp_mean", ascending=False)
        n = len(sub)
        if n == 0:
            continue
        risks = 1.0 - sub["miou"].values  # risk = 1 - mIoU
        coverages = np.arange(1, n + 1) / n
        cum_risk = np.cumsum(risks) / np.arange(1, n + 1)
        aurc = np.trapz(cum_risk, coverages)
        ax.plot(coverages, cum_risk, label=f"{m} (AURC={aurc:.4f})")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Cumulative Risk (1 - mIoU)")
    ax.set_title("Risk-Coverage Curve")
    ax.legend(fontsize=8)

    # 6. Inference time
    ax = axes[1, 2]
    for m in models:
        sub = combined[combined["model_name"] == m]
        ax.hist(sub["inference_ms"], bins=30, alpha=0.5, label=m, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Inference Latency")
    ax.legend(fontsize=8)

    plt.tight_layout()

    if save_dir:
        out = Path(save_dir) / "eval_overview.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved overview: {out}")

    if show:
        plt.show()
    plt.close(fig)

    # Additional: summary table
    if save_dir:
        summary_rows = []
        for m in models:
            sub = combined[combined["model_name"] == m]
            n = len(sub)
            risks = 1.0 - sub.sort_values("msp_mean", ascending=False)["miou"].values
            coverages = np.arange(1, n + 1) / n
            cum_risk = np.cumsum(risks) / np.arange(1, n + 1)
            aurc = np.trapz(cum_risk, coverages) if n > 0 else 0

            summary_rows.append({
                "model": m,
                "n_images": n,
                "miou_mean": sub["miou"].mean(),
                "miou_std": sub["miou"].std(),
                "pixel_acc": sub["pixel_acc"].mean(),
                "msp_mean": sub["msp_mean"].mean(),
                "entropy_mean": sub["entropy_mean"].mean(),
                "aurc": aurc,
                "avg_inference_ms": sub["inference_ms"].mean(),
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = Path(save_dir) / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[plot] Saved summary: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Guardrail Distillation Eval Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    # eval
    ep = sub.add_parser("eval", help="Run evaluation")
    ep.add_argument("--model", required=True, help="Path to model checkpoint")
    ep.add_argument("--dataset", required=True, help="Local path or hf://org/name[/split]")
    ep.add_argument("--output", default="results/eval.csv", help="Output CSV path")
    ep.add_argument("--num-classes", type=int, default=19)
    ep.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ep.add_argument("--max-samples", type=int, default=None)
    ep.add_argument("--model-name", default=None)
    ep.add_argument("--dataset-name", default=None)
    ep.add_argument("--hf-image-key", default="image")
    ep.add_argument("--hf-label-key", default="label")
    ep.add_argument("--hf-split", default=None)

    # plot
    pp = sub.add_parser("plot", help="Generate plots from CSV(s)")
    pp.add_argument("--csvs", nargs="+", required=True, help="One or more CSV files")
    pp.add_argument("--save-dir", default="results/figures")
    pp.add_argument("--show", action="store_true")

    args = parser.parse_args()

    if args.cmd == "eval":
        run_eval(
            model_path=args.model,
            dataset_path=args.dataset,
            output_csv=args.output,
            num_classes=args.num_classes,
            device=args.device,
            max_samples=args.max_samples,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            hf_image_key=args.hf_image_key,
            hf_label_key=args.hf_label_key,
            hf_split=args.hf_split,
        )
    elif args.cmd == "plot":
        plot_results(args.csvs, save_dir=args.save_dir, show=args.show)
    else:
        parser.print_help()