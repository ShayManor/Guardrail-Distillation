"""Per-dataset Pareto: confident-failure AUROC (MSP>=0.85) vs inference latency (b1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import apply_style, load_table, savefig, pool_acdc_domain, DATASET_LABELS

DATASETS = ("city", "acdc", "idd", "bdd")
BACKBONE = "b1"
MSP_THRESHOLD = 0.85

# (display, auroc column, supervision row, color, marker)
METHODS = [
    ("MSP",         "msp_auroc",        "dense_multi",  "#4C72B0", "o"),
    ("MaxLogit",    "max_logit_auroc",  "dense_multi",  "#64B5CD", "s"),
    ("MC-Dropout",  "mc_dropout_auroc", "dense_multi",  "#C44E52", "^"),
    ("GT-head",     "guardrail_auroc",  "gt_disagree",  "#55A868", "D"),
    ("Teacher-head (ours)", "guardrail_auroc", "dense_multi", "#8172B3", "*"),
]


def median_ms(lat: pd.DataFrame, dataset: str, col: str) -> float:
    sub = lat[lat["dataset"] == dataset]
    v = pd.to_numeric(sub[col], errors="coerce").dropna()
    v = v[v > 0]
    return float(v.median()) if len(v) else np.nan


def cost_for_method(name: str, s: float, g_teach: float, g_gt: float, mc: float, t: float) -> float:
    if name == "MSP":
        return s
    if name == "MaxLogit":
        # Same student forward; only the reduction differs from MSP.
        return s
    if name == "MC-Dropout":
        return mc if np.isfinite(mc) else np.nan
    if name == "GT-head":
        return s + g_gt if np.isfinite(g_gt) else np.nan
    if name == "Teacher-head (ours)":
        return s + g_teach if np.isfinite(g_teach) else np.nan
    return np.nan


def cf_auroc(cf: pd.DataFrame, dataset: str, supervision: str, col: str) -> float:
    sub = cf[
        (cf["dataset"] == dataset)
        & (cf["supervision_type"] == supervision)
        & (np.isclose(cf["msp_threshold"].astype(float), MSP_THRESHOLD))
    ]
    v = pd.to_numeric(sub[col], errors="coerce").dropna()
    return float(v.mean()) if len(v) else np.nan


def pareto_front(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Indices on the lower-x / higher-y frontier, sorted by x."""
    order = np.argsort(xs)
    frontier = []
    best_y = -np.inf
    for idx in order:
        if ys[idx] > best_y:
            frontier.append(idx)
            best_y = ys[idx]
    return np.array(frontier, dtype=int)


def main() -> None:
    apply_style()

    pi = load_table("per_image")
    cf = load_table("confident_failures")

    pi = pi[pi["backbone"] == BACKBONE].copy()
    cf = cf[cf["backbone"] == BACKBONE].copy()
    pi = pool_acdc_domain(pi)
    cf = pool_acdc_domain(cf)

    lat_teach = pi[pi["supervision_type"] == "dense_multi"]
    lat_gt    = pi[pi["supervision_type"] == "gt_disagree"]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex=False, sharey=False)
    axes = axes.ravel()

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        s  = median_ms(lat_teach, ds, "student_latency_ms")
        g_teach = median_ms(lat_teach, ds, "guardrail_latency_ms")
        g_gt = median_ms(lat_gt, ds, "guardrail_latency_ms")
        mc = median_ms(lat_teach, ds, "mc_dropout_latency_ms")
        t  = median_ms(lat_teach, ds, "teacher_latency_ms")

        xs, ys, labels, colors, markers = [], [], [], [], []
        for name, auroc_col, sup_row, color, marker in METHODS:
            cost = cost_for_method(name, s, g_teach, g_gt, mc, t)
            auroc = cf_auroc(cf, ds, sup_row, auroc_col)
            if not (np.isfinite(cost) and np.isfinite(auroc)):
                continue
            xs.append(cost); ys.append(auroc)
            labels.append(name); colors.append(color); markers.append(marker)

        # Reference point: pay full teacher latency, get teacher accuracy.
        if np.isfinite(s) and np.isfinite(t):
            xs.append(s + t); ys.append(1.0)
            labels.append("Teacher deferral"); colors.append("#937860"); markers.append("P")

        xs = np.array(xs); ys = np.array(ys)

        front = pareto_front(xs, ys)
        ax.plot(xs[front], ys[front], color="#aaa", lw=1.0, ls="--", alpha=0.7, zorder=1)

        for x, y, lab, col, mk in zip(xs, ys, labels, colors, markers):
            size = 170 if "ours" in lab else 110
            ax.scatter([x], [y], s=size, c=col, marker=mk, edgecolor="#1a1a1a",
                       linewidths=0.7, zorder=3, label=lab)
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color="#333")

        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11)
        ax.set_xlabel("Inference latency per image (ms, b1 median)")
        ax.set_ylabel(f"CF-AUROC @ MSP $\\geq$ {MSP_THRESHOLD}")
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Compute–accuracy Pareto (mit-b1). Lower-left is worse, upper-left is better.",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    savefig(fig, "fig_pareto")


if __name__ == "__main__":
    main()
