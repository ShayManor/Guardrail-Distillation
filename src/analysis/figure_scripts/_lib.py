"""Shared figure-script helpers.

Reads from src/analysis/combined_all/ (the merged output of every paper run)
and writes PNGs into src/analysis/figures/. Either path can be overridden via
GD_COMBINED / GD_FIGURES env vars or the first two CLI args. Single-seed and
multi-seed inputs both work; std bands appear only when n>=2.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent

def combined_dir() -> Path:
    p = os.environ.get("GD_COMBINED")
    if p:
        return Path(p)
    return ANALYSIS_ROOT / "combined_all"

def figures_dir() -> Path:
    p = os.environ.get("GD_FIGURES")
    if p:
        out = Path(p)
    else:
        out = ANALYSIS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out

BACKBONE_COLORS = {
    "b0": "#1f77b4",
    "b1": "#ff7f0e",
    "b2": "#2ca02c",
}
BACKBONE_LABELS = {"b0": "mit-b0", "b1": "mit-b1", "b2": "mit-b2"}

METHOD_COLORS = {
    "msp":        "#4C72B0",
    "temp_msp":   "#55A868",
    "entropy":    "#DD8452",
    "mc_dropout": "#C44E52",
    "energy":     "#CCB974",
    "max_logit":  "#64B5CD",
    "guardrail":  "#8172B3",
    "dense_gap":  "#8172B3",
    "dense_bce":  "#937860",
    "oracle":     "#6e6e6e",
    "random":     "#bdbdbd",
}
METHOD_LABELS = {
    "msp":        "MSP",
    "temp_msp":   "Temp-MSP",
    "entropy":    "Entropy",
    "mc_dropout": "MC-Dropout",
    "energy":     "Energy",
    "max_logit":  "MaxLogit",
    "guardrail":  "Guardrail (ours)",
    "dense_gap":  "Guardrail gap (ours)",
    "dense_bce":  "Guardrail BCE (ours)",
    "oracle":     "Oracle",
    "random":     "Random",
}

SUPERVISION_COLORS = {
    "dense_multi":    "#8172B3",
    "dense_gap":      "#4C72B0",
    "dense_disagree": "#DD8452",
    "scalar_benefit": "#C44E52",
    "gt_disagree":    "#55A868",
    "gt_risk":        "#CCB974",
}
SUPERVISION_LABELS = {
    "dense_multi":    "Teacher (multi-task)",
    "dense_gap":      "Teacher (gap only)",
    "dense_disagree": "Teacher (disagree only)",
    "scalar_benefit": "Teacher (scalar)",
    "gt_disagree":    "GT (disagree)",
    "gt_risk":        "GT (risk)",
}

DATASET_LABELS = {
    "city": "Cityscapes-val (in-domain)",
    "acdc": "ACDC (weather/lighting shift)",
    "idd":  "IDD (India, semantic shift)",
    "bdd":  "BDD100K (geographic shift)",
}

ALL_DATASETS = ("city", "acdc", "idd", "bdd")


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "figure.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


_SEED_RE = re.compile(r"_s(?P<seed>\d+)_j\d+$")

def derive_seed(row_source_dir: str) -> int:
    if not isinstance(row_source_dir, str):
        return 42
    m = _SEED_RE.search(row_source_dir)
    return int(m.group("seed")) if m else 42


def load_table(name: str, cdir: Optional[Path] = None) -> pd.DataFrame:
    cdir = cdir or combined_dir()
    path = cdir / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "source_dir" in df.columns and "seed" not in df.columns:
        df["seed"] = df["source_dir"].map(derive_seed)
    if "seed" not in df.columns:
        df["seed"] = 42
    # Normalize known values
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].replace({"cityscapes": "city", "cs": "city"})
    # BDD/IDD are pure OOD — pin them to the same "all" pooled-domain
    # convention ACDC uses so downstream filtering treats them uniformly.
    if "dataset" in df.columns and "domain" in df.columns:
        ood_mask = df["dataset"].isin(("bdd", "idd"))
        df.loc[ood_mask, "domain"] = "all"
    # combined_all merges every paper_eval_* dir; pre-max_logit/energy_score
    # rows collide with newer rows on the same primary key. Keep the newest
    # row that has the post-hoc baseline columns populated.
    dup_keys = [c for c in ("dataset", "backbone", "supervision_type",
                            "seed", "domain", "image_id") if c in df.columns]
    if "image_id" in df.columns and len(dup_keys) >= 4:
        score_cols = [c for c in ("max_logit", "energy_score")
                      if c in df.columns]
        df["__hasnew"] = 1
        if score_cols:
            df["__hasnew"] = df[score_cols].notna().all(axis=1).astype(int)
        sort_cols = ["__hasnew"]
        if "source_dir" in df.columns:
            sort_cols.append("source_dir")
        df = (df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
                .drop_duplicates(subset=dup_keys, keep="first")
                .drop(columns=["__hasnew"])
                .reset_index(drop=True))
    return df


def mean_std_across_seeds(df: pd.DataFrame, group_keys: Iterable[str],
                          value_col: str) -> pd.DataFrame:
    """One row per group with mean/std/n. std is NaN when n == 1."""
    g = df.groupby(list(group_keys))[value_col]
    out = pd.DataFrame({
        "mean": g.mean(),
        "std": g.std(ddof=1),
        "n": g.count(),
    }).reset_index()
    return out


def available_backbones(df: pd.DataFrame) -> list:
    if "backbone" not in df.columns:
        return []
    return [b for b in ("b0", "b1", "b2") if b in set(df["backbone"].dropna().unique())]


def available_datasets(df: pd.DataFrame) -> list:
    """Datasets present in df, in canonical order."""
    if "dataset" not in df.columns:
        return []
    present = set(df["dataset"].dropna().unique())
    return [ds for ds in ALL_DATASETS if ds in present]


def pool_acdc_domain(df: pd.DataFrame) -> pd.DataFrame:
    """ACDC has both per-condition and pooled rows; keep only the pooled superset."""
    if "domain" not in df.columns:
        return df
    mask_acdc = df["dataset"] == "acdc"
    keep = df["domain"] == "all"
    return pd.concat([df[~mask_acdc], df[mask_acdc & keep]], ignore_index=True)


def filter_mode(df: pd.DataFrame, mode: str = "dense_multi") -> pd.DataFrame:
    if "supervision_type" not in df.columns:
        return df
    return df[df["supervision_type"] == mode].copy()


def dedupe_runs(df: pd.DataFrame, keys=("dataset","backbone","supervision_type","run_id")) -> pd.DataFrame:
    present = [k for k in keys if k in df.columns]
    if not present:
        return df
    return df.drop_duplicates(subset=present + (["image_id"] if "image_id" in df.columns else []))


def cf_auroc(sub: pd.DataFrame, score_col: str, thr: float,
             label_col: str = "student_risk",
             q: float = 0.20,
             higher_is_fail: bool = True) -> Optional[float]:
    """Confident-failure AUROC: among images with student_msp >= thr,
    label the top-q fraction by `label_col` as failures and score with `score_col`.
    Negates scores when higher_is_fail=False. NaN if too few confident images.
    """
    from sklearn.metrics import roc_auc_score
    if score_col not in sub.columns or label_col not in sub.columns:
        return np.nan
    s = sub[sub["student_msp"] >= thr]
    if len(s) < 20:
        return np.nan
    lbl_vals = pd.to_numeric(s[label_col], errors="coerce").to_numpy()
    if np.isnan(lbl_vals).all():
        return np.nan
    k = max(1, int(round(q * len(s))))
    cutoff = np.sort(lbl_vals)[-k]
    y = (lbl_vals >= cutoff).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    scores = pd.to_numeric(s[score_col], errors="coerce").to_numpy()
    if not higher_is_fail:
        scores = -scores
    if np.isnan(scores).any():
        mask = ~np.isnan(scores)
        scores, y = scores[mask], y[mask]
        if y.sum() == 0 or y.sum() == len(y) or len(y) < 20:
            return np.nan
    return float(roc_auc_score(y, scores))


def cf_auroc_stratified(sub: pd.DataFrame, score_col: str, thr: float,
                        group_col: str = "condition",
                        label_col: str = "student_risk",
                        q: float = 0.20,
                        higher_is_fail: bool = True) -> Optional[float]:
    """Like cf_auroc but the failure cutoff is per-group, not pooled.

    Why: on ACDC, 18 of 19 pooled MSP>=0.85 failures come from night, so the
    pooled AUROC degenerates into a condition classifier. Per-group cutoffs
    force a confident failure in fog to compete against a confident non-fail
    in night under the same pooled score ranking.
    """
    from sklearn.metrics import roc_auc_score
    if score_col not in sub.columns or label_col not in sub.columns:
        return np.nan
    if group_col not in sub.columns:
        return cf_auroc(sub, score_col, thr, label_col=label_col, q=q,
                        higher_is_fail=higher_is_fail)
    s = sub[sub["student_msp"] >= thr].dropna(subset=[score_col, label_col]).copy()
    if len(s) < 20:
        return np.nan
    risk = pd.to_numeric(s[label_col], errors="coerce")
    s["_risk"] = risk
    s["_cut"] = s.groupby(group_col)["_risk"].transform(lambda x: x.quantile(1.0 - q))
    y = (s["_risk"] >= s["_cut"]).astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    scores = pd.to_numeric(s[score_col], errors="coerce").to_numpy()
    if not higher_is_fail:
        scores = -scores
    if np.isnan(scores).any():
        mask = ~np.isnan(scores)
        scores, y = scores[mask], y[mask]
        if y.sum() == 0 or y.sum() == len(y) or len(y) < 20:
            return np.nan
    return float(roc_auc_score(y, scores))


def risk_coverage_curve(sub: pd.DataFrame, score_col: str,
                        higher_is_fail: bool = True,
                        n_points: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by score (most confident first), return (coverage, mean-risk-so-far)."""
    if score_col not in sub.columns:
        return np.array([]), np.array([])
    s = sub.dropna(subset=[score_col, "student_risk"])
    if len(s) < 10:
        return np.array([]), np.array([])
    scores = pd.to_numeric(s[score_col], errors="coerce").to_numpy()
    risks = pd.to_numeric(s["student_risk"], errors="coerce").to_numpy()
    # Sort so that most-confident (lowest failure score) is first
    if higher_is_fail:
        order = np.argsort(scores)
    else:
        order = np.argsort(-scores)
    sorted_risks = risks[order]
    cumrisks = np.cumsum(sorted_risks) / (np.arange(len(sorted_risks)) + 1)
    n = len(sorted_risks)
    covs = np.linspace(1.0 / n, 1.0, n_points)
    return covs, cumrisks[np.clip((covs * n).astype(int) - 1, 0, n - 1)]


def aurc(sub: pd.DataFrame, score_col: str, higher_is_fail: bool = True) -> float:
    covs, risks = risk_coverage_curve(sub, score_col, higher_is_fail, n_points=100)
    if len(covs) < 2:
        return float("nan")
    return float(np.trapezoid(risks, covs))


def compute_aurc_table(pi: pd.DataFrame) -> pd.DataFrame:
    """Tidy AURC per (dataset, backbone, supervision_type, seed, method)."""
    df = pi.copy()
    if "domain" in df.columns:
        mask_acdc = df["dataset"] == "acdc"
        keep = df["domain"] == "all"
        df = pd.concat([df[~mask_acdc], df[mask_acdc & keep]], ignore_index=True)
    group_keys = ["dataset", "backbone", "supervision_type", "seed"]
    methods = [
        ("msp",        "student_msp",                     False),
        ("temp_msp",   "temp_msp",                        False),
        ("mc_dropout", "mc_entropy",                      True),
        ("energy",     "energy_score",                    True),
        ("max_logit",  "max_logit",                       False),
        ("dense_gap",  "guardrailpp_utility_dense_gap",   True),
        ("dense_bce",  "guardrailpp_utility_dense_bce",   True),
        ("oracle",     "student_risk",                    True),
    ]
    rows = []
    for keys, sub in df.groupby(group_keys):
        for name, col, hi in methods:
            rows.append({
                **dict(zip(group_keys, keys)),
                "method": name,
                "aurc": aurc(sub, col, hi),
            })
    return pd.DataFrame(rows)


def compute_cf_auroc_table(pi: pd.DataFrame,
                           msp_thresholds=(0.0, 0.85, 0.90, 0.95, 0.97),
                           pool_acdc_all: bool = True) -> pd.DataFrame:
    """Tidy confident-failure AUROC table swept across MSP thresholds.

    Columns: dataset, backbone, supervision_type, seed, msp_threshold, method,
    auroc, n_confident. Uses the CW-TR top-20% failure definition.
    """
    df = pi.copy()
    if "domain" in df.columns and pool_acdc_all:
        mask_acdc = df["dataset"] == "acdc"
        keep = df["domain"] == "all"
        df = pd.concat([df[~mask_acdc], df[mask_acdc & keep]], ignore_index=True)

    group_keys = ["dataset", "backbone", "supervision_type", "seed"]
    methods = [
        ("msp",        "student_msp",                     False),
        ("temp_msp",   "temp_msp",                        False),
        ("mc_dropout", "mc_entropy",                      True),
        ("energy",     "energy_score",                    True),
        ("max_logit",  "max_logit",                       False),
        ("dense_gap",  "guardrailpp_utility_dense_gap",   True),
        ("dense_bce",  "guardrailpp_utility_dense_bce",   True),
    ]
    rows = []
    for keys, sub in df.groupby(group_keys):
        for thr in msp_thresholds:
            for name, col, hi in methods:
                a = cf_auroc(sub, col, thr, higher_is_fail=hi)
                rec = dict(zip(group_keys, keys))
                rec.update({"msp_threshold": thr, "method": name,
                            "auroc": a,
                            "n_confident": int((sub["student_msp"] >= thr).sum())})
                rows.append(rec)
    return pd.DataFrame(rows)


def savefig(fig, name: str, outdir: Optional[Path] = None) -> Path:
    outdir = outdir or figures_dir()
    path = outdir / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → saved {path}")
    return path
