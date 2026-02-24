"""
Guardrail Distillation Evaluation Pipeline

Evaluates segmentation models on local or HuggingFace-streamed datasets.
Outputs per-image and aggregate metrics to CSV with visualization utilities.

Usage:
    from eval_pipeline import run_eval, plot_results, get_worst_k

    run_eval(
        model_path="./checkpoints/student_kd",
        dataset_path="hf://org/dark_zurich/val",
        output_csv="results/kd_darkzurich.csv",
    )
    plot_results("results/kd_darkzurich.csv", save_dir="results/figures")

    worst = get_worst_k("results/kd_darkzurich.csv", k_percent=5)
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

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


@dataclass
class ImageMetrics:
    image_id: str = ""
    miou: float = 0.0
    pixel_acc: float = 0.0
    mean_acc: float = 0.0
    msp_mean: float = 0.0
    msp_std: float = 0.0
    entropy_mean: float = 0.0
    entropy_std: float = 0.0
    per_class_iou: str = ""
    num_pixels: int = 0
    num_ignored: int = 0
    inference_ms: float = 0.0


def compute_metrics(
    pred_logits: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    image_id: str = "",
    inference_ms: float = 0.0,
) -> ImageMetrics:
    """Compute segmentation + uncertainty metrics for a single image."""
    probs = F.softmax(pred_logits.squeeze(0), dim=0)
    pred = probs.argmax(dim=0)
    label = label.squeeze()

    valid = label != IGNORE_INDEX
    pred_v = pred[valid]
    label_v = label[valid]
    probs_v = probs[:, valid]

    per_class_iou = {}
    ious = []
    for c in range(num_classes):
        p_c, l_c = pred_v == c, label_v == c
        inter = (p_c & l_c).sum().item()
        union = (p_c | l_c).sum().item()
        if union > 0:
            iou = inter / union
            ious.append(iou)
            per_class_iou[c] = round(iou, 5)

    total = valid.sum().item()
    pixel_acc = (pred_v == label_v).sum().item() / max(total, 1)

    class_accs = []
    for c in range(num_classes):
        mask = label_v == c
        if mask.sum() > 0:
            class_accs.append((pred_v[mask] == c).float().mean().item())
    mean_acc = np.mean(class_accs) if class_accs else 0.0

    max_probs = probs_v.max(dim=0).values
    eps = 1e-8
    ent = -(probs_v * (probs_v + eps).log()).sum(dim=0)

    return ImageMetrics(
        image_id=image_id,
        miou=np.mean(ious) if ious else 0.0,
        pixel_acc=pixel_acc,
        mean_acc=mean_acc,
        msp_mean=max_probs.mean().item(),
        msp_std=max_probs.std().item(),
        entropy_mean=ent.mean().item(),
        entropy_std=ent.std().item(),
        per_class_iou=json.dumps(per_class_iou),
        num_pixels=total,
        num_ignored=int((~valid).sum().item()),
        inference_ms=inference_ms,
    )


def _is_hf_path(path: str) -> bool:
    return path.startswith("hf://") or path.startswith("huggingface://")


def _parse_hf_path(path: str) -> tuple[str, Optional[str]]:
    """'hf://org/dataset[/split]' -> (dataset_id, split|None)"""
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
    """Load from local directory with matching image/label filenames."""
    root = Path(path)
    img_dir = root / images_subdir
    lbl_dir = root / labels_subdir

    if not img_dir.exists():
        img_dir = root / "leftImg8bit"
        lbl_dir = root / "gtFine"

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Cannot find images in {root}. Expected '{images_subdir}/' or 'leftImg8bit/'."
        )

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


def load_model(
    model_path: str,
    device: torch.device,
    num_classes: int = 19,
    model_factory: Optional[Callable] = None,
) -> torch.nn.Module:
    """
    Load a segmentation model. Supports TorchScript (.pt/.ts), full nn.Module
    pickles, and state-dict checkpoints (requires model_factory).
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if path.suffix.lower() in (".pt", ".ts"):
        try:
            model = torch.jit.load(str(path), map_location=device)
            model.eval()
            return model
        except Exception:
            pass

    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval()

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
    return model.to(device).eval()


@torch.no_grad()
def run_eval(
    model_path: str,
    dataset_path: str,
    output_csv: str,
    num_classes: int = 19,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_factory: Optional[Callable] = None,
    max_samples: Optional[int] = None,
    image_transform: Optional[Callable] = None,
    label_transform: Optional[Callable] = None,
    hf_image_key: str = "image",
    hf_label_key: str = "label",
    hf_split: Optional[str] = None,
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Path:
    """Run evaluation and write per-image metrics to CSV. Returns output path."""
    dev = torch.device(device)
    img_tf = image_transform or DEFAULT_TRANSFORM
    lbl_tf = label_transform or LABEL_TRANSFORM

    print(f"[eval] Loading model from {model_path}")
    model = load_model(model_path, dev, num_classes, model_factory)

    if _is_hf_path(dataset_path):
        print(f"[eval] Streaming from HuggingFace: {dataset_path}")
        data_iter = load_hf_stream(dataset_path, hf_image_key, hf_label_key, hf_split, max_samples)
    else:
        print(f"[eval] Loading local dataset: {dataset_path}")
        data_iter = load_local_dataset(dataset_path, images_subdir, labels_subdir, max_samples)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mn = model_name or Path(model_path).stem
    dn = dataset_name or dataset_path
    results: list[ImageMetrics] = []
    total_time = 0.0

    for img_id, pil_img, pil_lbl in data_iter:
        img_tensor = img_tf(pil_img).unsqueeze(0).to(dev)
        lbl_np = np.array(lbl_tf(pil_lbl)).astype(np.int64)
        lbl_tensor = torch.from_numpy(lbl_np).long().to(dev)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(img_tensor)
        if isinstance(out, dict):
            logits = out.get("out") or out.get("logits") or next(iter(out.values()))
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        if logits.shape[-2:] != lbl_tensor.shape[-2:]:
            logits = F.interpolate(logits, size=lbl_tensor.shape[-2:], mode="bilinear", align_corners=False)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        m = compute_metrics(logits, lbl_tensor, num_classes, img_id, elapsed_ms)
        results.append(m)
        total_time += elapsed_ms

        if len(results) % 50 == 0:
            running_miou = np.mean([r.miou for r in results])
            print(f"  [{len(results)}] running mIoU={running_miou:.4f}  last={elapsed_ms:.1f}ms")

    count = len(results)
    if count == 0:
        print("[eval] WARNING: No images processed.")
        return out_path

    fieldnames = ["model_name", "dataset_name"] + list(ImageMetrics.__dataclass_fields__.keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["model_name"] = mn
            row["dataset_name"] = dn
            writer.writerow(row)

    agg_miou = np.mean([r.miou for r in results])
    agg_pacc = np.mean([r.pixel_acc for r in results])
    avg_ms = total_time / count

    print(f"\n  Model: {mn}  |  Dataset: {dn}  |  N={count}")
    print(f"  mIoU={agg_miou:.4f}  PixAcc={agg_pacc:.4f}  AvgInf={avg_ms:.1f}ms")
    print(f"  Output: {out_path}\n")

    return out_path


def get_worst_k(
    csv_path: str,
    k_percent: float = 5.0,
    sort_by: str = "miou",
    ascending: bool = True,
) -> "pd.DataFrame":
    """
    Return the bottom k% of images ranked by a metric (default: lowest mIoU).

    Args:
        csv_path: Path to eval CSV from run_eval.
        k_percent: Bottom percentage to return (e.g. 5.0 = worst 5%).
        sort_by: Column to rank by. Use "miou" for most wrong, "msp_mean" for
                 least confident, "entropy_mean" for highest uncertainty.
        ascending: True = lowest values first (worst mIoU). Set False for
                   metrics where higher = worse (e.g. entropy_mean).

    Returns:
        DataFrame of the worst-k% rows, sorted.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    n = max(1, int(len(df) * k_percent / 100.0))
    worst = df.head(n).copy()

    print(f"[worst_k] {n}/{len(df)} images (bottom {k_percent}% by {sort_by})")
    print(f"  {sort_by} range: [{worst[sort_by].min():.4f}, {worst[sort_by].max():.4f}]")
    print(f"  mean mIoU in subset: {worst['miou'].mean():.4f}")

    return worst


def plot_results(
    csv_path: str | list[str],
    save_dir: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Generate analysis plots from one or more eval CSVs.

    Plots: mIoU distribution, confidence vs mIoU, per-class IoU bars,
    entropy vs mIoU, risk-coverage curve (AURC), inference latency.
    Writes summary.csv alongside figures.
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

    ax = axes[0, 0]
    for m in models:
        sub = combined[combined["model_name"] == m]
        ax.hist(sub["miou"], bins=30, alpha=0.5, label=m, edgecolor="black", linewidth=0.5)
    ax.set(xlabel="mIoU", ylabel="Count", title="mIoU Distribution")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for m in models:
        sub = combined[combined["model_name"] == m]
        ax.scatter(sub["msp_mean"], sub["miou"], alpha=0.3, s=10, label=m)
    ax.set(xlabel="Mean Softmax Prob (Confidence)", ylabel="mIoU", title="Confidence vs mIoU")
    ax.legend(fontsize=8)

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
        offset = 0.2 * list(models).index(m)
        ax.bar([c + offset for c in classes], means, width=0.2, alpha=0.7, label=m)
    ax.set(xlabel="Class ID", ylabel="Mean IoU", title="Per-Class IoU")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for m in models:
        sub = combined[combined["model_name"] == m]
        ax.scatter(sub["entropy_mean"], sub["miou"], alpha=0.3, s=10, label=m)
    ax.set(xlabel="Mean Entropy", ylabel="mIoU", title="Entropy vs mIoU")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for m in models:
        sub = combined[combined["model_name"] == m].sort_values("msp_mean", ascending=False)
        n = len(sub)
        if n == 0:
            continue
        risks = 1.0 - sub["miou"].values
        coverages = np.arange(1, n + 1) / n
        cum_risk = np.cumsum(risks) / np.arange(1, n + 1)
        aurc = np.trapz(cum_risk, coverages)
        ax.plot(coverages, cum_risk, label=f"{m} (AURC={aurc:.4f})")
    ax.set(xlabel="Coverage", ylabel="Cumulative Risk (1 - mIoU)", title="Risk-Coverage Curve")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    for m in models:
        sub = combined[combined["model_name"] == m]
        ax.hist(sub["inference_ms"], bins=30, alpha=0.5, label=m, edgecolor="black", linewidth=0.5)
    ax.set(xlabel="Inference Time (ms)", ylabel="Count", title="Inference Latency")
    ax.legend(fontsize=8)

    plt.tight_layout()

    if save_dir:
        out = Path(save_dir) / "eval_overview.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved overview: {out}")

    if show:
        plt.show()
    plt.close(fig)

    if save_dir:
        import pandas as pd
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
        summary_path = Path(save_dir) / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"[plot] Saved summary: {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Guardrail Distillation Eval Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    ep = sub.add_parser("eval", help="Run evaluation")
    ep.add_argument("--model", required=True)
    ep.add_argument("--dataset", required=True, help="Local path or hf://org/name[/split]")
    ep.add_argument("--output", default="results/eval.csv")
    ep.add_argument("--num-classes", type=int, default=19)
    ep.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ep.add_argument("--max-samples", type=int, default=None)
    ep.add_argument("--model-name", default=None)
    ep.add_argument("--dataset-name", default=None)
    ep.add_argument("--hf-image-key", default="image")
    ep.add_argument("--hf-label-key", default="label")
    ep.add_argument("--hf-split", default=None)

    pp = sub.add_parser("plot", help="Generate plots from CSV(s)")
    pp.add_argument("--csvs", nargs="+", required=True)
    pp.add_argument("--save-dir", default="results/figures")
    pp.add_argument("--show", action="store_true")

    wp = sub.add_parser("worst", help="Get bottom k%% most wrong images")
    wp.add_argument("--csv", required=True)
    wp.add_argument("--k", type=float, default=5.0, help="Bottom k percent")
    wp.add_argument("--sort-by", default="miou")
    wp.add_argument("--descending", action="store_true", help="Higher = worse (e.g. entropy)")
    wp.add_argument("--output", default=None, help="Save worst-k to CSV")

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
    elif args.cmd == "worst":
        worst = get_worst_k(args.csv, k_percent=args.k, sort_by=args.sort_by, ascending=not args.descending)
        if args.output:
            worst.to_csv(args.output, index=False)
            print(f"[worst_k] Saved to {args.output}")
        else:
            print(worst[["image_id", "miou", "pixel_acc", "msp_mean", "entropy_mean"]].to_string())
    else:
        parser.print_help()