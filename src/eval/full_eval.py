#!/usr/bin/env python3
"""
Unified paper evaluation script for guardrail distillation / robotic reflexes.

What this script does
---------------------
1) Evaluates ONE trained student checkpoint on ONE dataset split.
2) Optionally evaluates a teacher checkpoint and a guardrail checkpoint.
3) Writes / upserts a set of CSVs that are directly useful for the paper.
4) On a later run, can aggregate all CSVs into publication-friendly plots.

Outputs (CSV)
-------------
- runs.csv
    One row per (student run, dataset split) with global summary metrics.
- per_image.csv
    One row per image with risk / confidence / latency / oracle / guardrail scores.
- per_class.csv
    One row per class for each run with global IoU and support.
- risk_coverage.csv
    One row per run x method x coverage.
- teacher_budget.csv
    One row per run x method x teacher_budget.
- calibration_bins.csv
    One row per run x method x bin.
- confident_failures.csv
    One row per run x MSP threshold x method metrics.
- latency_samples.csv
    One row per image containing raw latency samples.

Outputs (figures)
-----------------
- overview.png
- pareto_by_dataset.png
- budget_benefit_by_dataset.png
- budget_effective_miou_by_dataset.png
- aurc_by_dataset.png
- calibration_summary.png

Important adapter points
------------------------
This script is intentionally generic in the evaluation core, but model loading and
loader construction are project-specific. The default adapters below try to use the
same project conventions as your current eval_guardrail pipeline:
    - config.Config
    - data.build_dataloaders
    - models.GuardrailHead / HFSegModelWrapper
    - utils.load_checkpoint

If your repository differs, only edit these adapter functions:
    - build_eval_loader(...)
    - build_student_model(...)
    - build_teacher_model(...)
    - build_guardrail_model(...)

Example usage
-------------
Evaluate one run and append all CSVs:

python full_eval.py eval \
  --run-id cityscapes_val_b2_kd \
  --dataset-name cityscapes \
  --dataset-path /root/Guardrail-Distillation/data/cityscapes \
  --split val \
  --student-name student_kd \
  --student-backbone nvidia/mit-b2 \
  --student-ckpt outputs-mit-b2/student_kd.ckpt \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --guardrail-ckpt outputs-mit-b2/guardrail.ckpt \
  --guardrail-student-name student_kd \
  --temperature 2.0 \
  --mc-dropout-passes 8 \
  --output-dir paper_eval \
  --batch-size 4

After evaluating all runs, make summary plots:

python full_eval.py plots --output-dir paper_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for this script") from exc

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for this script") from exc

IGNORE_INDEX = 255
DEFAULT_BUDGETS = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]
DEFAULT_COVERAGES = [round(x, 2) for x in np.linspace(0.05, 1.0, 20)]
DEFAULT_CONF_THRESHOLDS = [0.85, 0.90, 0.95, 0.97]
EPS = 1e-8

# Near-field ROI: bottom fraction of image (closest to ego vehicle)
NEAR_FIELD_FRAC = 1.0 / 3.0

# Cityscapes dynamic class IDs (safety-critical moving objects)
# person=11, rider=12, car=13, truck=14, bus=15, motorcycle=17, bicycle=18
DYNAMIC_CLASS_IDS = {11, 12, 13, 14, 15, 17, 18}


# =============================================================================
# Small utilities
# =============================================================================


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_float_list(text: str | None, default: Sequence[float]) -> List[float]:
    if text is None or text.strip() == "":
        return list(default)
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) > 0 else 0.0


def safe_std(values: Sequence[float]) -> float:
    return float(np.std(values)) if len(values) > 0 else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values), q))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# =============================================================================
# CSV upsert helpers
# =============================================================================


def upsert_rows(csv_path: Path, rows: List[Dict[str, Any]], key_cols: Sequence[str]) -> None:
    """Append or replace rows in a CSV based on key columns."""
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    for key in key_cols:
        if key not in new_df.columns:
            raise ValueError(f"Missing key column '{key}' for upsert into {csv_path}")

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        missing_old = [c for c in key_cols if c not in old_df.columns]
        if missing_old:
            # schema changed; overwrite with only new rows is safer than a broken merge
            out_df = new_df
        else:
            merged_keys = new_df[key_cols].astype(str).agg("||".join, axis=1).tolist()
            old_keys = old_df[key_cols].astype(str).agg("||".join, axis=1)
            out_df = pd.concat([old_df.loc[~old_keys.isin(merged_keys)], new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df.to_csv(csv_path, index=False)


# =============================================================================
# Metrics
# =============================================================================


def image_miou(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> float:
    valid = gt != IGNORE_INDEX
    if int(valid.sum()) == 0:
        return 0.0
    pred_v, gt_v = pred[valid], gt[valid]
    ious: List[float] = []
    for c in range(num_classes):
        p_c = pred_v == c
        g_c = gt_v == c
        union = int((p_c | g_c).sum())
        if union == 0:
            continue
        inter = int((p_c & g_c).sum())
        ious.append(inter / max(union, 1))
    return float(np.mean(ious)) if ious else 0.0


def image_pixel_acc(pred: torch.Tensor, gt: torch.Tensor) -> float:
    valid = gt != IGNORE_INDEX
    if int(valid.sum()) == 0:
        return 0.0
    return float((pred[valid] == gt[valid]).float().mean().item())


def per_class_inter_union(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> List[Tuple[int, int]]:
    valid = gt != IGNORE_INDEX
    pred_v, gt_v = pred[valid], gt[valid]
    out: List[Tuple[int, int]] = []
    for c in range(num_classes):
        p = pred_v == c
        g = gt_v == c
        inter = int((p & g).sum())
        union = int((p | g).sum())
        out.append((inter, union))
    return out


def per_class_support(gt: torch.Tensor, num_classes: int) -> List[int]:
    valid = gt != IGNORE_INDEX
    gt_v = gt[valid]
    counts: List[int] = []
    for c in range(num_classes):
        counts.append(int((gt_v == c).sum()))
    return counts


def image_miou_roi(
    pred: torch.Tensor, gt: torch.Tensor, num_classes: int, roi_frac: float = NEAR_FIELD_FRAC,
) -> Tuple[float, int]:
    """Compute mIoU on the bottom `roi_frac` of the image (near-field / ego lane region).
    Returns (miou, n_valid_pixels_in_roi). Returns (0.0, 0) if no valid pixels."""
    H = gt.shape[0]
    roi_start = int(H * (1.0 - roi_frac))
    pred_roi = pred[roi_start:]
    gt_roi = gt[roi_start:]
    valid = gt_roi != IGNORE_INDEX
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0, 0
    pred_v, gt_v = pred_roi[valid], gt_roi[valid]
    ious: List[float] = []
    for c in range(num_classes):
        p_c = pred_v == c
        g_c = gt_v == c
        union = int((p_c | g_c).sum())
        if union == 0:
            continue
        inter = int((p_c & g_c).sum())
        ious.append(inter / max(union, 1))
    miou = float(np.mean(ious)) if ious else 0.0
    return miou, n_valid


def image_miou_dynamic(
    pred: torch.Tensor, gt: torch.Tensor, dynamic_ids: set = DYNAMIC_CLASS_IDS,
) -> Tuple[float, int]:
    """Compute mIoU restricted to dynamic / safety-critical classes only.
    Returns (miou, n_valid_pixels_with_dynamic_gt). Returns (0.0, 0) if none present."""
    valid = gt != IGNORE_INDEX
    pred_v, gt_v = pred[valid], gt[valid]
    # Restrict to pixels where gt is one of the dynamic classes
    dyn_mask = torch.zeros_like(gt_v, dtype=torch.bool)
    for c in dynamic_ids:
        dyn_mask |= (gt_v == c)
    n_valid = int(dyn_mask.sum())
    if n_valid == 0:
        return 0.0, 0
    pred_d, gt_d = pred_v[dyn_mask], gt_v[dyn_mask]
    ious: List[float] = []
    for c in dynamic_ids:
        p_c = pred_d == c
        g_c = gt_d == c
        union = int((p_c | g_c).sum())
        if union == 0:
            continue
        inter = int((p_c & g_c).sum())
        ious.append(inter / max(union, 1))
    miou = float(np.mean(ious)) if ious else 0.0
    return miou, n_valid


def compute_aurc(risks: np.ndarray, keep_scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Sort by descending keep-score (higher means 'safer to keep'), then compute cumulative mean risk.
    Lower AURC is better.
    """
    order = np.argsort(-keep_scores)
    risks_sorted = risks[order]
    n = len(risks_sorted)
    coverages = np.arange(1, n + 1) / n
    cum_risk = np.cumsum(risks_sorted) / np.arange(1, n + 1)
    aurc = float(np.trapezoid(cum_risk, coverages))
    return aurc, coverages, cum_risk


def risk_at_coverage(risks: np.ndarray, keep_scores: np.ndarray, coverage: float) -> float:
    order = np.argsort(-keep_scores)
    k = max(1, int(round(len(risks) * coverage)))
    return float(np.mean(risks[order[:k]]))


def make_calibration_bins(pred_quality: np.ndarray, actual_quality: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (pred_quality >= lo) & (pred_quality <= hi)
        else:
            mask = (pred_quality >= lo) & (pred_quality < hi)
        if int(mask.sum()) == 0:
            rows.append({
                "bin_idx": i,
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": 0,
                "pred_mean": np.nan,
                "actual_mean": np.nan,
                "abs_gap": np.nan,
            })
            continue
        pred_mean = float(np.mean(pred_quality[mask]))
        actual_mean = float(np.mean(actual_quality[mask]))
        rows.append({
            "bin_idx": i,
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": int(mask.sum()),
            "pred_mean": pred_mean,
            "actual_mean": actual_mean,
            "abs_gap": abs(pred_mean - actual_mean),
        })
    return rows


def expected_calibration_error(bin_rows: Sequence[Dict[str, Any]]) -> float:
    total = sum(int(r["count"]) for r in bin_rows)
    if total == 0:
        return 0.0
    ece = 0.0
    for r in bin_rows:
        c = int(r["count"])
        if c == 0 or pd.isna(r["abs_gap"]):
            continue
        ece += (c / total) * float(r["abs_gap"])
    return float(ece)


def compute_confident_failure_table(
    df_img: pd.DataFrame,
    thresholds: Sequence[float],
    score_cols: Dict[str, str],
    failure_label: str = "top20",
) -> List[Dict[str, Any]]:
    """
    Labels 'failure' among confident examples.
    failure_label:
      - 'median': risk > median risk
      - 'top20': risk in worst 20%
      - 'top10': risk in worst 10%
    score_cols maps method name -> column where HIGHER means 'more likely to fail'.
    """
    if df_img.empty:
        return []

    if failure_label == "median":
        cutoff = float(df_img["student_risk"].median())
    elif failure_label == "top10":
        cutoff = float(df_img["student_risk"].quantile(0.90))
    else:  # default top20
        cutoff = float(df_img["student_risk"].quantile(0.80))

    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        conf = df_img[df_img["student_msp"] >= thr].copy()
        if len(conf) < 10:
            continue
        labels = (conf["student_risk"] >= cutoff).astype(int).values
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        row: Dict[str, Any] = {
            "msp_threshold": float(thr),
            "n_confident": int(len(conf)),
            "n_failures": int(labels.sum()),
            "failure_rate": float(labels.mean()),
            "failure_cutoff": cutoff,
            "failure_label": failure_label,
        }
        for method, col in score_cols.items():
            if col not in conf.columns:
                continue
            scores = conf[col].astype(float).values
            try:
                row[f"{method}_auroc"] = float(roc_auc_score(labels, scores))
                row[f"{method}_ap"] = float(average_precision_score(labels, scores))
            except ValueError:
                row[f"{method}_auroc"] = 0.5
                row[f"{method}_ap"] = float(labels.mean())
        rows.append(row)
    return rows


# =============================================================================
# Adapter hooks: edit only these if your project structure differs.
# =============================================================================


@dataclass
class EvalConfig:
    dataset_name: str
    dataset_path: str
    split: str
    domain: str
    batch_size: int
    num_workers: int
    num_classes: int
    device: str
    student_backbone: str
    teacher_backbone: Optional[str]
    temperature: float


class ForwardAdapter(nn.Module):
    """Wraps a project model so the rest of the script can assume logits output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, return_features: bool = False):
        out = self.model(x) if not return_features else self.model(x, return_features=True)
        return out


def build_eval_loader(cfg: EvalConfig):
    """
    Build validation DataLoader.
    - cityscapes: delegates to project's build_dataloaders
    - acdc: builds ACDCDataset directly (fog/night/rain/snow/all)
    """
    if cfg.dataset_name == "acdc":
        return _build_acdc_loader(cfg)

    # ── Cityscapes (default) ─────────────────────────────────────────────
    try:
        from src.train.config import Config
        from src.train.data import build_dataloaders
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import your project's Config / build_dataloaders. "
            "Edit build_eval_loader(...) in full_eval.py to match your repo."
        ) from exc

    project_cfg = Config(
        dataset_path=cfg.dataset_path,
        num_classes=cfg.num_classes,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        output_dir="unused-eval",
        device=cfg.device,
        lr=5e-5,
        epochs_sup=1,
        epochs_kd=1,
        epochs_skd=1,
        epochs_guardrail=1,
        eval_every=1,
        alpha_kd=0.5,
        alpha_struct=0.5,
        kd_temperature=cfg.temperature,
        log_every=100,
    )

    if hasattr(project_cfg, "dataset_name"):
        setattr(project_cfg, "dataset_name", cfg.dataset_name)
    if hasattr(project_cfg, "split"):
        setattr(project_cfg, "split", cfg.split)

    _, val_loader = build_dataloaders(project_cfg)
    return val_loader


# ── ACDC dataset & loader ────────────────────────────────────────────────────

ACDC_CONDITIONS = ("fog", "night", "rain", "snow")


def _build_acdc_loader(cfg: EvalConfig):
    """Build a DataLoader for ACDC adverse-conditions dataset."""
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

    # Map domain arg to condition filter
    condition = cfg.domain if cfg.domain in ACDC_CONDITIONS else "all"

    root = Path(cfg.dataset_path)
    conditions = ACDC_CONDITIONS if condition == "all" else (condition,)

    images, labels, conds = [], [], []
    uses_raw_label_ids = False  # True if falling back to _gt_labelIds.png
    for cond in conditions:
        img_dir = root / "rgb_anon" / cond / cfg.split
        lbl_dir = root / "gt" / cond / cfg.split

        if not img_dir.exists():
            print(f"[ACDC] WARNING: {img_dir} not found, skipping")
            continue

        for img_path in sorted(img_dir.rglob("*_rgb_anon.png")):
            rel = img_path.relative_to(img_dir)
            # labelTrainIds already mapped to 0-18 + 255
            lbl_name = img_path.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
            lbl_path = lbl_dir / rel.parent / lbl_name
            if not lbl_path.exists():
                # Fallback to _gt_labelIds.png (raw Cityscapes IDs, needs mapping)
                lbl_name = img_path.name.replace("_rgb_anon.png", "_gt_labelIds.png")
                lbl_path = lbl_dir / rel.parent / lbl_name
                if not lbl_path.exists():
                    continue
                uses_raw_label_ids = True
            images.append(str(img_path))
            labels.append(str(lbl_path))
            conds.append(cond)

    if not images:
        raise FileNotFoundError(
            f"No ACDC images found at {root}/rgb_anon/*/{ cfg.split}/\n"
            f"Run: bash setup_acdc.sh ./data/acdc /path/to/extracted/acdc"
        )

    per_cond = {c: conds.count(c) for c in conditions}
    print(f"[ACDC] {cfg.split} condition={condition}: {len(images)} images "
          f"({', '.join(f'{c}={n}' for c, n in per_cond.items())})")
    if uses_raw_label_ids:
        print("[ACDC] WARNING: Using _gt_labelIds.png (raw IDs) — applying Cityscapes label mapping")

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Cityscapes raw-ID → trainId LUT (only needed if labelTrainIds unavailable)
    _acdc_label_lut = None
    if uses_raw_label_ids:
        try:
            from src.train.data import _LABEL_LUT
            _acdc_label_lut = _LABEL_LUT
        except ImportError:
            # Inline Cityscapes label map: raw_id → train_id
            _CS_MAP = {
                7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
                21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                28: 15, 31: 16, 32: 17, 33: 18,
            }
            _acdc_label_lut = np.full(256, 255, dtype=np.uint8)
            for k, v in _CS_MAP.items():
                _acdc_label_lut[k] = v

    class _ACDCDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls, conds, img_size=(512, 512), label_lut=None):
            self.images = imgs
            self.labels = lbls
            self.conditions = conds
            self.img_size = img_size
            self.label_lut = label_lut  # None if labelTrainIds, LUT if raw labelIds

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = Image.open(self.images[idx]).convert("RGB")
            lbl = Image.open(self.labels[idx])

            if self.img_size:
                img = TF.resize(img, self.img_size,
                                interpolation=TF.InterpolationMode.BILINEAR)
                lbl = TF.resize(lbl, self.img_size,
                                interpolation=TF.InterpolationMode.NEAREST)

            img = TF.to_tensor(img)
            img = normalize(img)

            lbl_np = np.array(lbl, dtype=np.uint8)
            if self.label_lut is not None:
                lbl_np = self.label_lut[lbl_np]
            lbl = torch.from_numpy(lbl_np.astype(np.int64)).long()

            meta = {
                "image_id": Path(self.images[idx]).stem,
                "path": self.images[idx],
                "condition": self.conditions[idx],
            }
            return img, lbl, meta

    ds = _ACDCDataset(images, labels, conds, img_size=(512, 512),
                      label_lut=_acdc_label_lut)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_student_model(cfg: EvalConfig, checkpoint_path: str) -> nn.Module:
    try:
        from transformers import (
            AutoModelForSemanticSegmentation,
            SegformerConfig,
            SegformerForSemanticSegmentation,
        )
        from src.train.models import HFSegModelWrapper
        from src.train.utils import load_checkpoint
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import your project's student model helpers. "
            "Edit build_student_model(...) in full_eval.py to match your repo."
        ) from exc

    backbone = AutoModelForSemanticSegmentation.from_pretrained(cfg.student_backbone, local_files_only=True)
    s_cfg = SegformerConfig.from_pretrained(cfg.student_backbone, local_files_only=True)
    s_cfg.num_labels = cfg.num_classes
    model = SegformerForSemanticSegmentation(s_cfg)
    # Reuse encoder weights from backbone; head shape follows num_classes.
    model.segformer.load_state_dict(backbone.base_model.state_dict(), strict=False)
    wrapped = HFSegModelWrapper(model, cfg.num_classes)
    load_checkpoint(wrapped, checkpoint_path, device=cfg.device)
    return ForwardAdapter(wrapped).to(cfg.device).eval()


def build_teacher_model(cfg: EvalConfig) -> Optional[nn.Module]:
    if not cfg.teacher_backbone:
        return None
    try:
        from transformers import AutoModelForSemanticSegmentation
        from src.train.models import HFSegModelWrapper
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import your project's teacher model helpers. "
            "Edit build_teacher_model(...) in full_eval.py to match your repo."
        ) from exc

    raw = AutoModelForSemanticSegmentation.from_pretrained(cfg.teacher_backbone, local_files_only=True)
    wrapped = HFSegModelWrapper(raw, cfg.num_classes)
    return ForwardAdapter(wrapped).to(cfg.device).eval()


def build_guardrail_model(cfg: EvalConfig, checkpoint_path: Optional[str]) -> Optional[nn.Module]:
    if not checkpoint_path:
        return None
    try:
        from src.train.models import GuardrailPlusHead
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import GuardrailPlusHead from src.train.models."
        ) from exc

    state = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    enc_weight = state_dict["encoder.0.weight"]
    feat_ch = enc_weight.shape[1] - cfg.num_classes

    supervision_type = (
        state.get("supervision_type", "dense_multi") if isinstance(state, dict) else "dense_multi"
    )
    use_student_features_ckpt = (
        bool(state.get("use_student_features", feat_ch > 0))
        if isinstance(state, dict) else (feat_ch > 0)
    )
    print(
        f"[guardrail] feat_channels={feat_ch} supervision_type={supervision_type} "
        f"use_student_features={use_student_features_ckpt}"
    )

    model = GuardrailPlusHead(num_classes=cfg.num_classes, feat_channels=feat_ch)
    model.load_state_dict(state_dict)
    model = model.to(cfg.device).eval()
    # Stash metadata for downstream logging.
    model._supervision_type = supervision_type  # type: ignore[attr-defined]
    model._use_student_features = use_student_features_ckpt  # type: ignore[attr-defined]
    return model


# =============================================================================
# Batch normalization / forward helpers
# =============================================================================


def unpack_batch(batch: Any, batch_idx: int, base_index: int) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    if isinstance(batch, dict):
        if "image" in batch:
            images = batch["image"]
        else:
            images = batch["images"]
        if "label" in batch:
            labels = batch["label"]
        else:
            labels = batch["labels"]
        metas = batch.get("meta") or batch.get("metas") or []
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            images, labels = batch
            metas = []
        elif len(batch) >= 3:
            images, labels, metas = batch[0], batch[1], batch[2]
        else:
            raise ValueError("Unsupported batch tuple length")
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    bsz = int(images.shape[0])
    meta_list: List[Dict[str, Any]] = []
    if isinstance(metas, dict):
        # batch dict of lists
        for i in range(bsz):
            row = {}
            for k, v in metas.items():
                if isinstance(v, (list, tuple)) and len(v) == bsz:
                    row[k] = v[i]
                else:
                    row[k] = v
            meta_list.append(row)
    elif isinstance(metas, (list, tuple)) and len(metas) == bsz:
        for m in metas:
            meta_list.append(m if isinstance(m, dict) else {"meta": str(m)})
    else:
        for i in range(bsz):
            meta_list.append({})

    for i in range(bsz):
        meta_list[i].setdefault("image_id", f"img_{base_index + i:06d}")
        meta_list[i].setdefault("batch_idx", batch_idx)
        meta_list[i].setdefault("item_idx", i)

    return images, labels, meta_list


class Timer:
    def __init__(self, device: str):
        self.device = device
        self.start = 0.0
        self.end = 0.0

    def __enter__(self):
        cuda_sync(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        cuda_sync(self.device)
        self.end = time.perf_counter()

    @property
    def ms(self) -> float:
        return (self.end - self.start) * 1000.0


def enable_dropout_only(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


# =============================================================================
# Core evaluation: one run -> rows for all CSVs
# =============================================================================


def evaluate_one_run(args: argparse.Namespace) -> None:
    cfg = EvalConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=args.split,
        domain=args.domain,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        device=args.device,
        student_backbone=args.student_backbone,
        teacher_backbone=args.teacher_backbone,
        temperature=args.temperature,
    )

    out_root = ensure_dir(args.output_dir)
    csv_dir = ensure_dir(out_root / "csv")
    fig_dir = ensure_dir(out_root / "figures")

    budgets = parse_float_list(args.teacher_budgets, DEFAULT_BUDGETS)
    coverages = parse_float_list(args.coverages, DEFAULT_COVERAGES)
    conf_thresholds = parse_float_list(args.confident_thresholds, DEFAULT_CONF_THRESHOLDS)

    set_seed(args.seed)

    print(f"[eval] run_id={args.run_id}")
    print(f"[eval] dataset={args.dataset_name} split={args.split}")
    print(f"[eval] student={args.student_name} backbone={args.student_backbone}")

    loader = build_eval_loader(cfg)

    # Try to extract image file paths from the dataset for per-image tracking
    _dataset_paths: List[str] = []
    try:
        ds = loader.dataset
        for attr in ("images", "img_files", "image_paths", "filenames", "imgs"):
            if hasattr(ds, attr):
                _dataset_paths = [str(p) for p in getattr(ds, attr)]
                break
    except Exception:
        pass

    student = build_student_model(cfg, args.student_ckpt)
    teacher = build_teacher_model(cfg)
    guardrail = build_guardrail_model(cfg, args.guardrail_ckpt)
    guardrail_expects_feat = False
    if guardrail is not None and args.guardrail_ckpt:
        state = torch.load(args.guardrail_ckpt, map_location=cfg.device, weights_only=False)
        enc_w = state["model"]["encoder.0.weight"] if "model" in state else state["encoder.0.weight"]
        guardrail_expects_feat = (enc_w.shape[1] > cfg.num_classes)

    student_params = count_parameters(student)
    teacher_params = count_parameters(teacher) if teacher is not None else 0
    guardrail_params = count_parameters(guardrail) if guardrail is not None else 0

    # Warmup
    warm_batches = max(0, int(args.warmup_batches))
    if warm_batches > 0:
        print(f"[eval] warmup for {warm_batches} batches")
        for bi, batch in enumerate(loader):
            if bi >= warm_batches:
                break
            images, labels, _ = unpack_batch(batch, bi, bi * cfg.batch_size)
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            _ = student(images)
            if teacher is not None:
                _ = teacher(images)
            if guardrail is not None:
                try:
                    out = student(images, return_features=True)
                except TypeError:
                    out = student(images)
                if isinstance(out, tuple):
                    logits, feat = out
                else:
                    logits, feat = out, None
                feat = None
                if isinstance(out, tuple) and len(out) > 1:
                    feat = out[1]
                _wfeat = feat if guardrail_expects_feat else None
                try:
                    _ = guardrail(logits, _wfeat)
                except TypeError:
                    _ = guardrail(logits)

    per_image_rows: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, Any]] = []
    class_inter = np.zeros(cfg.num_classes, dtype=np.int64)
    class_union = np.zeros(cfg.num_classes, dtype=np.int64)
    class_support = np.zeros(cfg.num_classes, dtype=np.int64)
    teacher_class_inter = np.zeros(cfg.num_classes, dtype=np.int64)
    teacher_class_union = np.zeros(cfg.num_classes, dtype=np.int64)

    seen_images = 0
    eval_image_res = ""  # will be set from first batch

    # Main loop
    student.eval()
    if teacher is not None:
        teacher.eval()
    if guardrail is not None:
        guardrail.eval()

    for batch_idx, batch in enumerate(loader):
        images, labels, metas = unpack_batch(batch, batch_idx, seen_images)
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)
        bsz = int(images.shape[0])

        # Capture eval image resolution from first batch
        if not eval_image_res:
            eval_image_res = f"{images.shape[2]}x{images.shape[3]}"

        # Student forward
        with Timer(cfg.device) as t_student:
            try:
                out = student(images, return_features=True)
            except TypeError:
                out = student(images)
        if isinstance(out, tuple):
            student_logits = out[0]
            student_feat = out[1] if len(out) > 1 else None
        else:
            student_logits = out
            student_feat = None
        student_ms_img = t_student.ms / max(bsz, 1)

        # Temperature-scaled logits (for confidence only)
        temp_logits = student_logits / max(args.temperature, EPS)

        # Teacher forward
        teacher_logits = None
        teacher_ms_img = 0.0
        if teacher is not None:
            with Timer(cfg.device) as t_teacher:
                teacher_out = teacher(images)
            teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out
            teacher_ms_img = t_teacher.ms / max(bsz, 1)

        # Guardrail forward
        guard_raw = None
        guard_ms_img = 0.0
        adapt_logits = None  # kept for schema compatibility; no adaptive preprocessing now
        if guardrail is not None and args.guardrail_student_name == args.student_name:
            _guard_feat = student_feat if guardrail_expects_feat else None
            with Timer(cfg.device) as t_guard:
                try:
                    guard_raw = guardrail(student_logits, _guard_feat)
                except TypeError:
                    guard_raw = guardrail(student_logits)
            guard_ms_img = t_guard.ms / max(bsz, 1)

        # MC dropout forward (uncertainty only)
        mc_entropies: Optional[List[float]] = None
        mc_mutual_infos: Optional[List[float]] = None
        if args.mc_dropout_passes > 0:
            probs_samples: List[torch.Tensor] = []
            student.eval()
            enable_dropout_only(student)
            with Timer(cfg.device) as t_mc:
                with torch.no_grad():
                    for _ in range(int(args.mc_dropout_passes)):
                        mc_out = student(images)
                        mc_logits = mc_out[0] if isinstance(mc_out, tuple) else mc_out
                        probs_samples.append(F.softmax(mc_logits, dim=1))
            student.eval()
            probs_stack = torch.stack(probs_samples, dim=0)  # [T, B, C, H, W]
            probs_mean = probs_stack.mean(dim=0)
            pred_entropy = -(probs_mean * (probs_mean + EPS).log()).sum(dim=1)
            exp_entropy = -(
                probs_stack * (probs_stack + EPS).log()
            ).sum(dim=2).mean(dim=0)
            mutual_info = pred_entropy - exp_entropy
            mc_entropies = []
            mc_mutual_infos = []
            for i in range(bsz):
                valid = labels[i] != IGNORE_INDEX
                if int(valid.sum()) == 0:
                    mc_entropies.append(0.0)
                    mc_mutual_infos.append(0.0)
                else:
                    mc_entropies.append(float(pred_entropy[i][valid].mean().item()))
                    mc_mutual_infos.append(float(mutual_info[i][valid].mean().item()))
            mc_latency_per_img = t_mc.ms / max(bsz, 1)

        student_probs = F.softmax(student_logits, dim=1)
        temp_probs = F.softmax(temp_logits, dim=1)
        student_preds = student_logits.argmax(dim=1)
        teacher_preds = teacher_logits.argmax(dim=1) if teacher_logits is not None else None

        for i in range(bsz):
            meta = metas[i]
            image_id = str(meta.get("image_id", f"img_{seen_images + i:06d}"))
            image_path = str(meta.get("path", meta.get("file_name", meta.get("filename", ""))))
            if not image_path and (seen_images + i) < len(_dataset_paths):
                image_path = _dataset_paths[seen_images + i]
            valid = labels[i] != IGNORE_INDEX
            if int(valid.sum()) == 0:
                continue

            pred = student_preds[i]
            gt = labels[i]
            student_miou = image_miou(pred, gt, cfg.num_classes)
            student_risk = 1.0 - student_miou
            student_acc = image_pixel_acc(pred, gt)

            # Near-field ROI risk (bottom 1/3 of image)
            student_miou_near, n_near = image_miou_roi(pred, gt, cfg.num_classes)
            student_risk_near = 1.0 - student_miou_near

            # Dynamic class risk (person, rider, car, truck, bus, motorcycle, bicycle)
            student_miou_dynamic, n_dynamic = image_miou_dynamic(pred, gt)
            student_risk_dynamic = 1.0 - student_miou_dynamic

            inter_union = per_class_inter_union(pred, gt, cfg.num_classes)
            support = per_class_support(gt, cfg.num_classes)
            for cls_idx, (inter, union) in enumerate(inter_union):
                class_inter[cls_idx] += inter
                class_union[cls_idx] += union
                class_support[cls_idx] += support[cls_idx]

            probs_valid = student_probs[i][:, valid]
            temp_probs_valid = temp_probs[i][:, valid]

            pixel_max = probs_valid.max(dim=0).values
            pixel_max_temp = temp_probs_valid.max(dim=0).values
            pixel_ent = -(probs_valid * (probs_valid + EPS).log()).sum(dim=0)
            pixel_ent_temp = -(temp_probs_valid * (temp_probs_valid + EPS).log()).sum(dim=0)

            row: Dict[str, Any] = {
                "run_id": args.run_id,
                "timestamp": now_ts(),
                "dataset_name": args.dataset_name,
                "split": args.split,
                "domain": args.domain,
                "student_name": args.student_name,
                "student_backbone": args.student_backbone,
                "student_ckpt": args.student_ckpt,
                "train_method": args.train_method,
                "image_id": image_id,
                "image_path": image_path,
                "condition": meta.get("condition", ""),
                "student_miou": student_miou,
                "student_risk": student_risk,
                "student_pixel_acc": student_acc,
                "student_msp": float(pixel_max.mean().item()),
                "student_msp_std": float(pixel_max.std().item()) if pixel_max.numel() > 1 else 0.0,
                "student_entropy": float(pixel_ent.mean().item()),
                "student_entropy_std": float(pixel_ent.std().item()) if pixel_ent.numel() > 1 else 0.0,
                "temp_msp": float(pixel_max_temp.mean().item()),
                "temp_entropy": float(pixel_ent_temp.mean().item()),
                "low_conf_frac_050": float((pixel_max < 0.50).float().mean().item()),
                "low_conf_frac_070": float((pixel_max < 0.70).float().mean().item()),
                "n_valid_pixels": int(valid.sum().item()),
                "student_latency_ms": float(student_ms_img),
                "guardrail_latency_ms": float(guard_ms_img),
                "teacher_latency_ms": float(teacher_ms_img),
                "student_miou_near": student_miou_near,
                "student_risk_near": student_risk_near,
                "n_near_pixels": n_near,
                "student_miou_dynamic": student_miou_dynamic,
                "student_risk_dynamic": student_risk_dynamic,
                "n_dynamic_pixels": n_dynamic,
            }

            if mc_entropies is not None and mc_mutual_infos is not None:
                row["mc_entropy"] = float(mc_entropies[i])
                row["mc_mutual_info"] = float(mc_mutual_infos[i])
                row["mc_dropout_latency_ms"] = float(mc_latency_per_img)

            if guard_raw is not None and isinstance(guard_raw, dict):
                # ── Dense disagreement head (primary in dense_multi mode) ──
                if "disagree_logits" in guard_raw:
                    dl = guard_raw["disagree_logits"][i]
                    if dl.ndim == 2 and dl.shape == gt.shape:
                        dl_valid = torch.sigmoid(dl[valid])
                    else:
                        dl_valid = torch.sigmoid(dl.flatten())
                    if dl_valid.numel() > 0:
                        util_bce = float(dl_valid.mean().item())
                        row["guardrailpp_utility_dense_bce"] = util_bce
                        row["guardrailpp_keep_dense_bce"] = 1.0 - util_bce

                # ── Dense signed risk-gap head (primary alternative) ──
                if "gap_pred" in guard_raw:
                    gp = guard_raw["gap_pred"][i]
                    if gp.ndim == 2 and gp.shape == gt.shape:
                        gp_valid = gp[valid]
                    else:
                        gp_valid = gp.flatten()
                    if gp_valid.numel() > 0:
                        util_gap_raw = float(gp_valid.mean().item())
                        row["guardrailpp_utility_dense_gap_raw"] = util_gap_raw
                        util_gap = float(
                            torch.sigmoid(torch.tensor(util_gap_raw)).item()
                        )
                        row["guardrailpp_utility_dense_gap"] = util_gap
                        row["guardrailpp_keep_dense_gap"] = 1.0 - util_gap

                # ── Scalar utility head (only trained under scalar_benefit
                #     supervision; kept for the ablation) ──
                if "utility_score" in guard_raw:
                    utility = float(guard_raw["utility_score"][i].item())
                    utility = max(0.0, min(1.0, utility))
                    row["guardrailpp_utility_scalar"] = utility

                # ── Primary `guardrailpp_utility` alias used by plots /
                #     legacy tables: picks dense_gap if present, else
                #     dense_bce, else scalar. Never the dead scalar head. ──
                if "guardrailpp_utility_dense_gap" in row:
                    primary = row["guardrailpp_utility_dense_gap"]
                elif "guardrailpp_utility_dense_bce" in row:
                    primary = row["guardrailpp_utility_dense_bce"]
                elif "guardrailpp_utility_scalar" in row:
                    primary = row["guardrailpp_utility_scalar"]
                else:
                    primary = None
                if primary is not None:
                    row["guardrailpp_utility"] = primary
                    row["guardrailpp_keep"] = 1.0 - primary
                    row["guardrail_risk"] = primary
                    row["guardrail_keep"] = 1.0 - primary

            if teacher_preds is not None:
                t_pred = teacher_preds[i]
                teacher_miou = image_miou(t_pred, gt, cfg.num_classes)
                teacher_risk = 1.0 - teacher_miou
                teacher_acc = image_pixel_acc(t_pred, gt)

                # Accumulate teacher per-class stats
                t_inter_union = per_class_inter_union(t_pred, gt, cfg.num_classes)
                for cls_idx, (t_inter, t_union) in enumerate(t_inter_union):
                    teacher_class_inter[cls_idx] += t_inter
                    teacher_class_union[cls_idx] += t_union

                # Teacher ROI and dynamic metrics
                teacher_miou_near, _ = image_miou_roi(t_pred, gt, cfg.num_classes)
                teacher_risk_near = 1.0 - teacher_miou_near
                teacher_miou_dynamic, _ = image_miou_dynamic(t_pred, gt)
                teacher_risk_dynamic = 1.0 - teacher_miou_dynamic

                student_correct = pred[valid] == gt[valid]
                teacher_correct = t_pred[valid] == gt[valid]
                teacher_better_mask = teacher_correct & (~student_correct)
                student_better_mask = student_correct & (~teacher_correct)
                disagreement_mask = pred[valid] != t_pred[valid]
                confident_mask = pixel_max >= 0.90
                cwt_mask = confident_mask & (~student_correct) & teacher_correct

                row.update({
                    "teacher_miou": teacher_miou,
                    "teacher_risk": teacher_risk,
                    "teacher_pixel_acc": teacher_acc,
                    "teacher_miou_near": teacher_miou_near,
                    "teacher_risk_near": teacher_risk_near,
                    "teacher_benefit_near": max(student_risk_near - teacher_risk_near, 0.0),
                    "teacher_miou_dynamic": teacher_miou_dynamic,
                    "teacher_risk_dynamic": teacher_risk_dynamic,
                    "teacher_benefit_dynamic": max(student_risk_dynamic - teacher_risk_dynamic, 0.0),
                    "teacher_gain": max(student_miou - teacher_miou, 0.0) * -1.0,  # overwritten below for readability
                    "teacher_benefit": max(student_risk - teacher_risk, 0.0),
                    "teacher_gap": float(teacher_better_mask.float().mean().item()),
                    "student_better_gap": float(student_better_mask.float().mean().item()),
                    "oracle_keep": 1.0 - max(student_risk - teacher_risk, 0.0),
                    "oracle_fail": max(student_risk - teacher_risk, 0.0),
                    "disagreement_rate": float(disagreement_mask.float().mean().item()),
                    "confident_wrong_teacher_right": float(cwt_mask.float().mean().item()) if int(confident_mask.sum()) > 0 else 0.0,
                    "n_confident_pixels": int(confident_mask.sum().item()),
                })
                row["teacher_gain"] = row["teacher_benefit"]

            per_image_rows.append(row)
            latency_rows.append({
                "run_id": args.run_id,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "student_name": args.student_name,
                "image_id": image_id,
                "student_latency_ms": float(student_ms_img),
                "guardrail_latency_ms": float(guard_ms_img),
                "teacher_latency_ms": float(teacher_ms_img),
            })

        seen_images += bsz
        if seen_images % max(args.progress_every, 1) < bsz:
            print(f"[eval] processed {seen_images} images")

    if not per_image_rows:
        raise RuntimeError("No per-image rows were produced. Check your dataloader and model adapter.")

    df_img = pd.DataFrame(per_image_rows)

    # -------------------------------------------------------------------------
    # Per-class dataset rows
    # -------------------------------------------------------------------------
    class_rows: List[Dict[str, Any]] = []
    for c in range(cfg.num_classes):
        iou = float(class_inter[c] / class_union[c]) if class_union[c] > 0 else np.nan
        t_iou = float(teacher_class_inter[c] / teacher_class_union[c]) if teacher_class_union[c] > 0 else np.nan
        row_cls: Dict[str, Any] = {
            "run_id": args.run_id,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "domain": args.domain,
            "student_name": args.student_name,
            "student_backbone": args.student_backbone,
            "train_method": args.train_method,
            "class_id": c,
            "class_name": args.class_names[c] if args.class_names and c < len(args.class_names) else f"class_{c}",
            "iou": iou,
            "support_pixels": int(class_support[c]),
            "inter_pixels": int(class_inter[c]),
            "union_pixels": int(class_union[c]),
            "teacher_iou": t_iou,
            "teacher_inter_pixels": int(teacher_class_inter[c]),
            "teacher_union_pixels": int(teacher_class_union[c]),
        }
        class_rows.append(row_cls)

    # -------------------------------------------------------------------------
    # Risk-coverage rows
    # -------------------------------------------------------------------------
    score_keep_map: Dict[str, np.ndarray] = {
        "msp": df_img["student_msp"].values,
        "neg_entropy": (-df_img["student_entropy"].values),
        "temp_msp": df_img["temp_msp"].values,
    }
    if "mc_entropy" in df_img.columns:
        score_keep_map["mc_dropout"] = -df_img["mc_entropy"].values
    if "guardrail_keep" in df_img.columns:
        score_keep_map["guardrail"] = df_img["guardrail_keep"].values
    if "guardrailpp_keep" in df_img.columns:
        score_keep_map["guardrailpp_keep"] = df_img["guardrailpp_keep"].values
    if "oracle_keep" in df_img.columns:
        score_keep_map["teacher_oracle"] = df_img["oracle_keep"].values
    # True selective-prediction oracle: rank by actual quality (ground truth at test time)
    score_keep_map["oracle"] = df_img["student_miou"].values

    risk_cov_rows: List[Dict[str, Any]] = []
    student_risk_arr = df_img["student_risk"].values.astype(float)
    for method, keep_scores in score_keep_map.items():
        aurc, curve_cov, curve_risk = compute_aurc(student_risk_arr, keep_scores)
        curve_lookup = {round(float(c), 4): float(r) for c, r in zip(curve_cov, curve_risk)}
        for cov in coverages:
            risk_cov_rows.append({
                "run_id": args.run_id,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "domain": args.domain,
                "student_name": args.student_name,
                "student_backbone": args.student_backbone,
                "train_method": args.train_method,
                "method": method,
                "coverage": float(cov),
                "risk": risk_at_coverage(student_risk_arr, keep_scores, cov),
                "aurc": float(aurc),
            })

    # -------------------------------------------------------------------------
    # Teacher-budget rows (effective system performance under fallback)
    # -------------------------------------------------------------------------
    budget_rows: List[Dict[str, Any]] = []
    teacher_budget_methods = {
        "msp": 1.0 - df_img["student_msp"].values,  # high = more likely to fallback
        "entropy": df_img["student_entropy"].values,
        "temp_msp": 1.0 - df_img["temp_msp"].values,
    }
    if "mc_entropy" in df_img.columns:
        teacher_budget_methods["mc_dropout"] = df_img["mc_entropy"].values
    if "guardrail_risk" in df_img.columns:
        teacher_budget_methods["guardrail"] = df_img["guardrail_risk"].values
    if "guardrailpp_utility" in df_img.columns:
        teacher_budget_methods["guardrailpp_utility"] = df_img["guardrailpp_utility"].values
    if "oracle_fail" in df_img.columns:
        teacher_budget_methods["oracle"] = df_img["oracle_fail"].values
    teacher_budget_methods["random"] = np.random.RandomState(args.seed).rand(len(df_img))

    # If calibrated utility available, add λ-sweep method
    calib_path = Path(args.output_dir) / "guardrail_calibrator.pkl"
    if "guardrailpp_utility" in df_img.columns and calib_path.exists():
        import pickle
        with open(calib_path, "rb") as f:
            iso = pickle.load(f)
        calibrated_utility = iso.predict(df_img["guardrailpp_utility"].values.astype(float))
        df_img["calibrated_utility"] = calibrated_utility
        teacher_budget_methods["guardrailpp_calibrated"] = calibrated_utility

    teacher_available = "teacher_risk" in df_img.columns
    total_teacher_benefit = float(df_img["teacher_benefit"].sum()) if teacher_available else 0.0

    for method, fail_scores in teacher_budget_methods.items():
        order = np.argsort(-fail_scores)  # highest failure score => fallback first
        n = len(order)
        for budget in budgets:
            k = int(round(budget * n))
            selected = np.zeros(n, dtype=bool)
            if k > 0:
                selected[order[:k]] = True
            invocation_rate = float(selected.mean())

            eff_risk = student_risk_arr.copy()
            eff_miou = df_img["student_miou"].values.astype(float).copy()
            benefit_recovered = 0.0
            teacher_on_selected_risk = np.zeros(n, dtype=float)
            if teacher_available:
                teacher_risk_arr = df_img["teacher_risk"].values.astype(float)
                teacher_miou_arr = df_img["teacher_miou"].values.astype(float)
                benefit_arr = df_img["teacher_benefit"].values.astype(float)
                eff_risk[selected] = teacher_risk_arr[selected]
                eff_miou[selected] = teacher_miou_arr[selected]
                benefit_recovered = float(benefit_arr[selected].sum())
                teacher_on_selected_risk[selected] = teacher_risk_arr[selected]

            student_lat = df_img["student_latency_ms"].values.astype(float)
            guard_lat = df_img["guardrail_latency_ms"].values.astype(float) if "guardrail_latency_ms" in df_img else np.zeros(n)
            teacher_lat = df_img["teacher_latency_ms"].values.astype(float) if "teacher_latency_ms" in df_img else np.zeros(n)

            method_guard_cost = guard_lat if method in ("guardrail", "guardrailpp_utility") else np.zeros(n)
            mc_lat = df_img["mc_dropout_latency_ms"].values.astype(float) if "mc_dropout_latency_ms" in df_img else np.zeros(n)
            method_mc_cost = mc_lat if method == "mc_dropout" else np.zeros(n)
            total_lat = student_lat + method_guard_cost + method_mc_cost + selected.astype(float) * teacher_lat

            budget_rows.append({
                "run_id": args.run_id,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "domain": args.domain,
                "student_name": args.student_name,
                "student_backbone": args.student_backbone,
                "train_method": args.train_method,
                "method": method,
                "teacher_budget": float(budget),
                "teacher_invocation_rate": invocation_rate,
                "effective_risk": float(np.mean(eff_risk)),
                "effective_miou": float(np.mean(eff_miou)),
                "avg_latency_ms": float(np.mean(total_lat)),
                "p95_latency_ms": percentile(total_lat, 95),
                "p99_latency_ms": percentile(total_lat, 99),
                "benefit_recovered": benefit_recovered,
                "benefit_recovered_frac": float(benefit_recovered / max(total_teacher_benefit, EPS)) if teacher_available else 0.0,
                "total_teacher_benefit": total_teacher_benefit,
            })

    # -------------------------------------------------------------------------
    # Calibration rows
    # -------------------------------------------------------------------------
    calib_rows: List[Dict[str, Any]] = []
    actual_quality = 1.0 - df_img["student_risk"].values.astype(float)
    calib_sources: Dict[str, np.ndarray] = {
        "msp": df_img["student_msp"].values.astype(float),
        "temp_msp": df_img["temp_msp"].values.astype(float),
    }
    if "guardrail_keep" in df_img.columns:
        calib_sources["guardrail"] = df_img["guardrail_keep"].values.astype(float)

    run_ece: Dict[str, float] = {}
    for method, pred_quality in calib_sources.items():
        bins = make_calibration_bins(pred_quality, actual_quality, n_bins=args.calibration_bins)
        run_ece[method] = expected_calibration_error(bins)
        for b in bins:
            calib_rows.append({
                "run_id": args.run_id,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "domain": args.domain,
                "student_name": args.student_name,
                "student_backbone": args.student_backbone,
                "train_method": args.train_method,
                "method": method,
                **b,
            })

    # -------------------------------------------------------------------------
    # Confident failure rows
    # -------------------------------------------------------------------------
    fail_score_cols = {
        "msp": 1.0 - df_img["student_msp"].values.astype(float),
        "entropy": df_img["student_entropy"].values.astype(float),
        "temp_msp": 1.0 - df_img["temp_msp"].values.astype(float),
    }
    df_fail = df_img.copy()
    df_fail["msp_fail_score"] = fail_score_cols["msp"]
    df_fail["entropy_fail_score"] = fail_score_cols["entropy"]
    df_fail["temp_msp_fail_score"] = fail_score_cols["temp_msp"]
    score_cols = {
        "msp": "msp_fail_score",
        "entropy": "entropy_fail_score",
        "temp_msp": "temp_msp_fail_score",
    }
    if "mc_entropy" in df_fail.columns:
        df_fail["mc_dropout_fail_score"] = df_fail["mc_entropy"].astype(float)
        score_cols["mc_dropout"] = "mc_dropout_fail_score"
    # `guardrail_risk` is aliased to the primary dense utility (dense_gap
    # fallback dense_bce fallback scalar) so this row always uses the trained
    # column when available.
    if "guardrail_risk" in df_fail.columns:
        score_cols["guardrail"] = "guardrail_risk"
    # Register the dense heads as their own rows so the CSV carries both in
    # addition to the aliased "guardrail" row.
    if "guardrailpp_utility_dense_bce" in df_fail.columns:
        score_cols["dense_bce"] = "guardrailpp_utility_dense_bce"
    if "guardrailpp_utility_dense_gap" in df_fail.columns:
        score_cols["dense_gap"] = "guardrailpp_utility_dense_gap"
    if "oracle_fail" in df_fail.columns:
        score_cols["oracle"] = "oracle_fail"

    confident_rows_raw = compute_confident_failure_table(df_fail, conf_thresholds, score_cols, failure_label=args.failure_label)
    confident_rows: List[Dict[str, Any]] = []
    for r in confident_rows_raw:
        confident_rows.append({
            "run_id": args.run_id,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "domain": args.domain,
            "student_name": args.student_name,
            "student_backbone": args.student_backbone,
            "train_method": args.train_method,
            **r,
        })

    # -------------------------------------------------------------------------
    # Summary run row
    # -------------------------------------------------------------------------
    run_row: Dict[str, Any] = {
        "run_id": args.run_id,
        "timestamp": now_ts(),
        "seed": int(args.seed),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.version.cuda else "",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "eval_batch_size": int(args.batch_size),
        "eval_image_res": eval_image_res,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "domain": args.domain,
        "student_name": args.student_name,
        "student_backbone": args.student_backbone,
        "student_ckpt": args.student_ckpt,
        "train_method": args.train_method,
        "teacher_backbone": args.teacher_backbone or "",
        "guardrail_ckpt": args.guardrail_ckpt or "",
        "guardrail_student_name": args.guardrail_student_name or "",
        "guardrail_supervision_type": (
            getattr(guardrail, "_supervision_type", "") if guardrail is not None else ""
        ),
        "guardrail_use_student_features": (
            int(getattr(guardrail, "_use_student_features", False))
            if guardrail is not None else 0
        ),
        "num_images": int(len(df_img)),
        "num_classes": cfg.num_classes,
        "temperature": float(args.temperature),
        "mc_dropout_passes": int(args.mc_dropout_passes),
        "student_params": int(student_params),
        "teacher_params": int(teacher_params),
        "guardrail_params": int(guardrail_params),
        "student_miou": float(df_img["student_miou"].mean()),
        "student_risk": float(df_img["student_risk"].mean()),
        "student_pixel_acc": float(df_img["student_pixel_acc"].mean()),
        "student_msp": float(df_img["student_msp"].mean()),
        "student_entropy": float(df_img["student_entropy"].mean()),
        "student_risk_near": float(df_img["student_risk_near"].mean()),
        "student_miou_near": float(df_img["student_miou_near"].mean()),
        "student_risk_dynamic": float(df_img["student_risk_dynamic"].mean()),
        "student_miou_dynamic": float(df_img["student_miou_dynamic"].mean()),
        "student_latency_ms": float(df_img["student_latency_ms"].mean()),
        "student_latency_p95_ms": percentile(df_img["student_latency_ms"].values, 95),
        "guardrail_latency_ms": float(df_img["guardrail_latency_ms"].mean()) if "guardrail_latency_ms" in df_img else 0.0,
        "teacher_latency_ms": float(df_img["teacher_latency_ms"].mean()) if "teacher_latency_ms" in df_img else 0.0,
        "msp_aurc": float(next(r["aurc"] for r in risk_cov_rows if r["method"] == "msp")),
        "entropy_aurc": float(next(r["aurc"] for r in risk_cov_rows if r["method"] == "neg_entropy")),
        "temp_msp_aurc": float(next(r["aurc"] for r in risk_cov_rows if r["method"] == "temp_msp")),
        "msp_ece": float(run_ece.get("msp", 0.0)),
        "temp_msp_ece": float(run_ece.get("temp_msp", 0.0)),
    }
    if "mc_entropy" in df_img.columns:
        run_row["mc_dropout_aurc"] = float(next(r["aurc"] for r in risk_cov_rows if r["method"] == "mc_dropout"))
        run_row["mc_entropy"] = float(df_img["mc_entropy"].mean())
    if "guardrail_risk" in df_img.columns:
        run_row["guardrail_aurc"] = float(next(r["aurc"] for r in risk_cov_rows if r["method"] == "guardrail"))
        run_row["guardrail_risk"] = float(df_img["guardrail_risk"].mean())
        run_row["guardrail_ece"] = float(run_ece.get("guardrail", 0.0))
        corr = np.corrcoef(df_img["guardrail_risk"].values.astype(float), df_img["student_risk"].values.astype(float))[0, 1]
        run_row["guardrail_vs_risk_corr"] = float(corr) if not math.isnan(corr) else 0.0
    if "teacher_miou" in df_img.columns:
        run_row["teacher_miou"] = float(df_img["teacher_miou"].mean())
        run_row["teacher_risk"] = float(df_img["teacher_risk"].mean())
        run_row["teacher_pixel_acc"] = float(df_img["teacher_pixel_acc"].mean())
        run_row["teacher_benefit"] = float(df_img["teacher_benefit"].mean())
        if "teacher_risk_near" in df_img.columns:
            run_row["teacher_risk_near"] = float(df_img["teacher_risk_near"].mean())
            run_row["teacher_benefit_near"] = float(df_img["teacher_benefit_near"].mean())
            run_row["teacher_risk_dynamic"] = float(df_img["teacher_risk_dynamic"].mean())
            run_row["teacher_benefit_dynamic"] = float(df_img["teacher_benefit_dynamic"].mean())
        run_row["oracle_aurc"] = float(next(r["aurc"] for r in risk_cov_rows if r["method"] == "oracle"))
        if any(r["method"] == "teacher_oracle" for r in risk_cov_rows):
            run_row["teacher_oracle_aurc"] = float(next(r["aurc"] for r in risk_cov_rows if r["method"] == "teacher_oracle"))
        run_row["disagreement_rate"] = float(df_img["disagreement_rate"].mean())
        run_row["confident_wrong_teacher_right"] = float(df_img["confident_wrong_teacher_right"].mean())
        for budget in [0.01, 0.05, 0.10, 0.20]:
            subset = [r for r in budget_rows if abs(r["teacher_budget"] - budget) < 1e-9]
            for method in ["msp", "entropy", "temp_msp", "mc_dropout", "guardrail", "oracle"]:
                match = [r for r in subset if r["method"] == method]
                if match:
                    run_row[f"{method}_benefit_at_{int(budget*100)}"] = float(match[0]["benefit_recovered_frac"])
                    run_row[f"{method}_miou_at_{int(budget*100)}"] = float(match[0]["effective_miou"])
                    run_row[f"{method}_lat_at_{int(budget*100)}"] = float(match[0]["avg_latency_ms"])

    # -------------------------------------------------------------------------
    # Save / upsert all CSVs
    # -------------------------------------------------------------------------
    upsert_rows(csv_dir / "runs.csv", [run_row], key_cols=["run_id"])
    upsert_rows(csv_dir / "per_image.csv", per_image_rows, key_cols=["run_id", "image_id"])
    upsert_rows(csv_dir / "per_class.csv", class_rows, key_cols=["run_id", "class_id"])
    upsert_rows(csv_dir / "risk_coverage.csv", risk_cov_rows, key_cols=["run_id", "method", "coverage"])
    upsert_rows(csv_dir / "teacher_budget.csv", budget_rows, key_cols=["run_id", "method", "teacher_budget"])
    upsert_rows(csv_dir / "calibration_bins.csv", calib_rows, key_cols=["run_id", "method", "bin_idx"])
    upsert_rows(csv_dir / "confident_failures.csv", confident_rows, key_cols=["run_id", "msp_threshold"])
    upsert_rows(csv_dir / "latency_samples.csv", latency_rows, key_cols=["run_id", "image_id"])

    print(f"[done] wrote CSVs under {csv_dir}")

    # Make one quick pretty chart for this run immediately.
    quick_plot_for_run(
        df_img=df_img,
        risk_cov_rows=pd.DataFrame(risk_cov_rows),
        budget_rows=pd.DataFrame(budget_rows),
        confident_rows=pd.DataFrame(confident_rows),
        save_path=fig_dir / f"quicklook_{args.run_id}.png",
        title=args.run_id,
    )
    print(f"[done] quick plot -> {fig_dir / f'quicklook_{args.run_id}.png'}")


# =============================================================================
# Plotting
# =============================================================================


def quick_plot_for_run(
    df_img: pd.DataFrame,
    risk_cov_rows: pd.DataFrame,
    budget_rows: pd.DataFrame,
    confident_rows: pd.DataFrame,
    save_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(16, 10))

    # 1) MSP vs risk
    ax1 = plt.subplot(2, 2, 1)
    sc = ax1.scatter(df_img["student_msp"], df_img["student_risk"], c=df_img["student_risk"], s=10, alpha=0.5)
    ax1.set_title("MSP vs Image Risk")
    ax1.set_xlabel("MSP")
    ax1.set_ylabel("Risk (1 - mIoU)")
    ax1.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax1)

    # 2) Risk-coverage
    ax2 = plt.subplot(2, 2, 2)
    for method, sub in risk_cov_rows.groupby("method"):
        ax2.plot(sub["coverage"], sub["risk"], label=method)
    ax2.set_title("Risk-Coverage")
    ax2.set_xlabel("Coverage")
    ax2.set_ylabel("Risk")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # 3) Teacher budget -> effective miou
    ax3 = plt.subplot(2, 2, 3)
    for method, sub in budget_rows.groupby("method"):
        ax3.plot(sub["teacher_budget"], sub["effective_miou"], label=method)
    ax3.set_title("Teacher Budget vs Effective mIoU")
    ax3.set_xlabel("Teacher budget")
    ax3.set_ylabel("Effective mIoU")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # 4) Teacher budget -> avg latency
    ax4 = plt.subplot(2, 2, 4)
    for method, sub in budget_rows.groupby("method"):
        ax4.plot(sub["teacher_budget"], sub["avg_latency_ms"], label=method)
    ax4.set_title("Teacher Budget vs Avg Latency")
    ax4.set_xlabel("Teacher budget")
    ax4.set_ylabel("Avg latency (ms)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(output_dir: str) -> None:
    out_root = ensure_dir(output_dir)
    csv_dir = ensure_dir(out_root / "csv")
    fig_dir = ensure_dir(out_root / "figures")

    runs_csv = csv_dir / "runs.csv"
    rc_csv = csv_dir / "risk_coverage.csv"
    budget_csv = csv_dir / "teacher_budget.csv"
    calib_csv = csv_dir / "calibration_bins.csv"

    if not runs_csv.exists():
        raise FileNotFoundError(f"Missing {runs_csv}")
    if not rc_csv.exists():
        raise FileNotFoundError(f"Missing {rc_csv}")
    if not budget_csv.exists():
        raise FileNotFoundError(f"Missing {budget_csv}")

    df_runs = pd.read_csv(runs_csv)
    df_rc = pd.read_csv(rc_csv)
    df_budget = pd.read_csv(budget_csv)
    df_calib = pd.read_csv(calib_csv) if calib_csv.exists() else pd.DataFrame()

    # ------------------------------------------------------------------
    # Figure 1: Pareto by dataset (use 10% teacher budget as main op point)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, max(1, df_budget["dataset_name"].nunique()), figsize=(7 * max(1, df_budget["dataset_name"].nunique()), 5), squeeze=False)
    for ax, (dataset, sub) in zip(axes[0], df_budget.groupby("dataset_name")):
        op = sub[np.isclose(sub["teacher_budget"], 0.10)]
        for method, msub in op.groupby("method"):
            ax.scatter(msub["avg_latency_ms"], msub["effective_miou"], s=70, label=method, alpha=0.85)
            for _, r in msub.iterrows():
                ax.annotate(r["student_name"], (r["avg_latency_ms"], r["effective_miou"]), fontsize=8, alpha=0.8)
        ax.set_title(f"{dataset}: 10% teacher budget")
        ax.set_xlabel("Average latency (ms)")
        ax.set_ylabel("Effective mIoU")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "pareto_by_dataset.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Teacher budget -> benefit recovered by dataset
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, max(1, df_budget["dataset_name"].nunique()), figsize=(7 * max(1, df_budget["dataset_name"].nunique()), 5), squeeze=False)
    for ax, (dataset, sub) in zip(axes[0], df_budget.groupby("dataset_name")):
        agg = sub.groupby(["teacher_budget", "method"], as_index=False)["benefit_recovered_frac"].mean()
        for method, msub in agg.groupby("method"):
            ax.plot(msub["teacher_budget"], msub["benefit_recovered_frac"], label=method)
        ax.set_title(f"Teacher benefit recovered: {dataset}")
        ax.set_xlabel("Teacher budget")
        ax.set_ylabel("Recovered fraction of total teacher benefit")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "budget_benefit_by_dataset.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Teacher budget -> effective mIoU by dataset
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, max(1, df_budget["dataset_name"].nunique()), figsize=(7 * max(1, df_budget["dataset_name"].nunique()), 5), squeeze=False)
    for ax, (dataset, sub) in zip(axes[0], df_budget.groupby("dataset_name")):
        agg = sub.groupby(["teacher_budget", "method"], as_index=False)["effective_miou"].mean()
        for method, msub in agg.groupby("method"):
            ax.plot(msub["teacher_budget"], msub["effective_miou"], label=method)
        ax.set_title(f"Teacher budget vs effective mIoU: {dataset}")
        ax.set_xlabel("Teacher budget")
        ax.set_ylabel("Effective mIoU")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "budget_effective_miou_by_dataset.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: AURC by dataset / method
    # ------------------------------------------------------------------
    aurc_rows = df_rc.groupby(["dataset_name", "student_name", "method"], as_index=False)["aurc"].mean()
    n_ds = max(1, aurc_rows["dataset_name"].nunique())
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)
    for ax, (dataset, sub) in zip(axes[0], aurc_rows.groupby("dataset_name")):
        students = list(sub["student_name"].drop_duplicates())
        methods = list(sub["method"].drop_duplicates())
        x = np.arange(len(students))
        width = 0.8 / max(len(methods), 1)
        for i, method in enumerate(methods):
            vals = []
            for s in students:
                match = sub[(sub["student_name"] == s) & (sub["method"] == method)]
                vals.append(float(match["aurc"].iloc[0]) if len(match) else np.nan)
            ax.bar(x + i * width, vals, width, label=method, alpha=0.85)
        ax.set_xticks(x + width * max(len(methods) - 1, 0) / 2)
        ax.set_xticklabels(students, rotation=15)
        ax.set_title(f"AURC: {dataset}")
        ax.set_ylabel("AURC (lower = better)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "aurc_by_dataset.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: Calibration summary (predicted quality vs actual quality)
    # ------------------------------------------------------------------
    if not df_calib.empty:
        n_ds = max(1, df_calib["dataset_name"].nunique())
        fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)
        for ax, (dataset, sub) in zip(axes[0], df_calib.groupby("dataset_name")):
            for method, msub in sub.groupby("method"):
                msub = msub.sort_values("bin_idx")
                ax.plot(msub["pred_mean"], msub["actual_mean"], marker="o", label=method)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_title(f"Calibration: {dataset}")
            ax.set_xlabel("Predicted quality")
            ax.set_ylabel("Empirical quality")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(fig_dir / "calibration_summary.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 6: One combined overview image
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 2, 1)
    for dataset, sub in df_runs.groupby("dataset_name"):
        ax1.scatter(sub["student_latency_ms"], sub["student_miou"], s=80, alpha=0.8, label=dataset)
        for _, r in sub.iterrows():
            ax1.annotate(r["student_name"], (r["student_latency_ms"], r["student_miou"]), fontsize=8)
    ax1.set_title("Student-only accuracy vs latency")
    ax1.set_xlabel("Student latency (ms)")
    ax1.set_ylabel("Student mIoU")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = plt.subplot(2, 2, 2)
    op = df_budget[np.isclose(df_budget["teacher_budget"], 0.10)]
    for method, sub in op.groupby("method"):
        ax2.scatter(sub["avg_latency_ms"], sub["effective_miou"], s=80, alpha=0.8, label=method)
    ax2.set_title("10% teacher budget: effective mIoU vs latency")
    ax2.set_xlabel("Average latency (ms)")
    ax2.set_ylabel("Effective mIoU")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = plt.subplot(2, 2, 3)
    agg = df_budget.groupby(["teacher_budget", "method"], as_index=False)["benefit_recovered_frac"].mean()
    for method, sub in agg.groupby("method"):
        ax3.plot(sub["teacher_budget"], sub["benefit_recovered_frac"], label=method)
    ax3.set_title("Recovered teacher benefit")
    ax3.set_xlabel("Teacher budget")
    ax3.set_ylabel("Benefit recovered fraction")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    ax4 = plt.subplot(2, 2, 4)
    aurc_avg = df_rc.groupby(["method"], as_index=False)["aurc"].mean().sort_values("aurc")
    ax4.bar(aurc_avg["method"], aurc_avg["aurc"], alpha=0.85)
    ax4.set_title("Mean AURC across runs")
    ax4.set_ylabel("AURC (lower = better)")
    ax4.tick_params(axis="x", rotation=20)
    ax4.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] wrote figures under {fig_dir}")


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified paper evaluation for guardrail distillation.")
    sub = parser.add_subparsers(dest="mode", required=True)

    # eval mode
    p_eval = sub.add_parser("eval", help="Evaluate one trained student on one dataset split and append CSVs.")
    p_eval.add_argument("--run-id", required=True, help="Unique run key, e.g. cityscapes_val_b2_kd")
    p_eval.add_argument("--dataset-name", required=True, help="cityscapes | acdc | dark_zurich | ...")
    p_eval.add_argument("--dataset-path", required=True)
    p_eval.add_argument("--split", default="val")
    p_eval.add_argument("--domain", default="in_domain", help="in_domain | night | rain | fog | snow | ood")
    p_eval.add_argument("--student-name", required=True, help="student_sup | student_kd | student_skd | ...")
    p_eval.add_argument("--student-backbone", required=True, help="e.g. nvidia/mit-b2")
    p_eval.add_argument("--student-ckpt", required=True)
    p_eval.add_argument("--train-method", default="unknown", help="sup | kd | skd | finetune | ...")
    p_eval.add_argument("--teacher-backbone", default=None)
    p_eval.add_argument("--guardrail-ckpt", default=None)
    p_eval.add_argument("--guardrail-student-name", default="student_skd")
    p_eval.add_argument("--num-classes", type=int, default=19)
    p_eval.add_argument("--class-names", nargs="*", default=None)
    p_eval.add_argument("--batch-size", type=int, default=4)
    p_eval.add_argument("--num-workers", type=int, default=0)
    p_eval.add_argument("--temperature", type=float, default=2.0)
    p_eval.add_argument("--mc-dropout-passes", type=int, default=0)
    p_eval.add_argument("--teacher-budgets", default=None, help="Comma-separated list, e.g. 0,0.01,0.05,0.1,0.2,0.5,1.0")
    p_eval.add_argument("--coverages", default=None, help="Comma-separated list for risk-coverage output")
    p_eval.add_argument("--confident-thresholds", default=None, help="Comma-separated list, e.g. 0.9,0.95,0.97")
    p_eval.add_argument("--failure-label", default="top20", choices=["median", "top20", "top10"])
    p_eval.add_argument("--calibration-bins", type=int, default=10)
    p_eval.add_argument("--warmup-batches", type=int, default=5)
    p_eval.add_argument("--progress-every", type=int, default=50)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--seeds", default=None, help="Comma-separated seeds for multi-seed runs, e.g. 42,137,256. Overrides --seed.")
    p_eval.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_eval.add_argument("--output-dir", default="paper_eval")

    # plots mode
    p_plots = sub.add_parser("plots", help="Aggregate existing CSVs and make summary plots.")
    p_plots.add_argument("--output-dir", default="paper_eval")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "eval":
        if args.seeds:
            seed_list = [int(s.strip()) for s in args.seeds.split(",")]
            base_run_id = args.run_id
            for seed in seed_list:
                args.seed = seed
                args.run_id = f"{base_run_id}_s{seed}"
                print(f"\n{'='*60}\n[multi-seed] run_id={args.run_id}  seed={seed}\n{'='*60}")
                evaluate_one_run(args)
            args.run_id = base_run_id  # restore
        else:
            evaluate_one_run(args)
    elif args.mode == "plots":
        make_all_plots(args.output_dir)
    else:  # pragma: no cover
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
