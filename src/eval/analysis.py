"""Metrics computation, worst-k extraction, and visualization."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from data import IGNORE_INDEX


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
    """Segmentation + uncertainty metrics for a single image."""
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

    max_probs = probs_v.max(dim=0).values
    eps = 1e-8
    ent = -(probs_v * (probs_v + eps).log()).sum(dim=0)
    n_valid = max_probs.numel()

    return ImageMetrics(
        image_id=image_id,
        miou=np.mean(ious) if ious else 0.0,
        pixel_acc=pixel_acc,
        mean_acc=np.mean(class_accs) if class_accs else 0.0,
        msp_mean=max_probs.mean().item(),
        msp_std=max_probs.std().item() if n_valid > 1 else 0.0,
        entropy_mean=ent.mean().item(),
        entropy_std=ent.std().item() if n_valid > 1 else 0.0,
        per_class_iou=json.dumps(per_class_iou),
        num_pixels=total,
        num_ignored=int((~valid).sum().item()),
        inference_ms=inference_ms,
    )


def get_worst_k(csv_path: str, k_percent: float = 5.0, sort_by: str = "miou", ascending: bool = True):
    """
    Bottom k% of images by a metric. Default: lowest mIoU.
    Set ascending=False for metrics where higher = worse (e.g. entropy_mean).
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


def plot_results(csv_path: str | list[str], save_dir: Optional[str] = None, show: bool = False):
    """
    6-panel analysis from eval CSVs: mIoU dist, confidence vs mIoU, per-class IoU,
    entropy vs mIoU, risk-coverage (AURC), latency. Writes summary.csv.
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
    ax.set(xlabel="Mean Softmax Prob", ylabel="mIoU", title="Confidence vs mIoU")
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
        aurc = np.trapezoid(cum_risk, coverages)
        ax.plot(coverages, cum_risk, label=f"{m} (AURC={aurc:.4f})")
    ax.set(xlabel="Coverage", ylabel="Cumulative Risk", title="Risk-Coverage Curve")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    for m in models:
        sub = combined[combined["model_name"] == m]
        ax.hist(sub["inference_ms"], bins=30, alpha=0.5, label=m, edgecolor="black", linewidth=0.5)
    ax.set(xlabel="Inference Time (ms)", ylabel="Count", title="Inference Latency")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "eval_overview.png", dpi=150, bbox_inches="tight")
        print(f"[plot] Saved {Path(save_dir) / 'eval_overview.png'}")
    if show:
        plt.show()
    plt.close(fig)

    if save_dir:
        summary_rows = []
        for m in models:
            sub = combined[combined["model_name"] == m]
            n = len(sub)
            risks = 1.0 - sub.sort_values("msp_mean", ascending=False)["miou"].values
            coverages = np.arange(1, n + 1) / n
            cum_risk = np.cumsum(risks) / np.arange(1, n + 1)
            aurc = np.trapezoid(cum_risk, coverages) if n > 0 else 0
            summary_rows.append({
                "model": m, "n_images": n,
                "miou_mean": sub["miou"].mean(), "miou_std": sub["miou"].std(),
                "pixel_acc": sub["pixel_acc"].mean(), "msp_mean": sub["msp_mean"].mean(),
                "entropy_mean": sub["entropy_mean"].mean(), "aurc": aurc,
                "avg_inference_ms": sub["inference_ms"].mean(),
            })
        summary_path = Path(save_dir) / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"[plot] Saved {summary_path}")
