"""Supervision-mode ablation including GT baselines (b1 only).

Grouped bar chart: CF-AUROC @ msp>=0.85 across all supervision modes.
Shows that all teacher modes beat GT modes under OOD shift,
and all learned modes beat post-hoc baselines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _lib import (
    apply_style, load_table, savefig, cf_auroc, pool_acdc_domain,
    available_datasets, DATASET_LABELS, per_seed_apply,
)

MODES = ("dense_multi", "dense_gap", "dense_disagree", "scalar_benefit",
         "gt_disagree", "gt_risk")


def trained_col(mode):
    return {
        "dense_multi":    "guardrailpp_utility_dense_gap",
        "dense_gap":      "guardrailpp_utility_dense_gap",
        "dense_disagree": "guardrailpp_utility_dense_bce",
        "scalar_benefit": "guardrailpp_utility_scalar",
        "gt_disagree":    "guardrailpp_utility_dense_bce",
        "gt_risk":        "guardrailpp_utility_dense_gap",
    }[mode]


# Distinct, harmonious dataset colors. Cool grey for in-domain so OOD
# colors carry the emphasis.
DATASET_COLORS = {
    "city": "#9aa0a6",   # neutral grey  (in-domain)
    "acdc": "#3b6ea5",   # blue          (weather/lighting shift)
    "idd":  "#d97c2f",   # warm orange   (semantic shift)
    "bdd":  "#7a4ea0",   # purple        (geographic shift)
}

# Light tints for category background bands (pale, non-distracting).
CATEGORY_BANDS = [
    # (xmin, xmax, color, label)
    (-0.5, 0.5, "#f4f4f4", "Post-hoc\nbaseline"),
    (0.5, 1.5, "#fdecec", "Negative\nresult"),
    (1.5, 3.5, "#ecf5ec", "GT-supervised"),
    (3.5, 6.5, "#eaf0fa", "Teacher-supervised\n(ours)"),
]


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["backbone"] == "b1"].copy()
    pi = pool_acdc_domain(pi)

    datasets = available_datasets(pi)

    rows = []
    for ds in datasets:
        sub_ds = pi[pi["dataset"] == ds]
        sub_any = sub_ds[sub_ds["supervision_type"] == "dense_multi"]
        if sub_any.empty:
            sub_any = sub_ds.drop_duplicates(subset=["image_id"])
        m, s, n = per_seed_apply(sub_any,
            lambda g: cf_auroc(g, "student_msp", 0.85, higher_is_fail=False))
        rows.append({"mode": "MSP", "dataset": ds, "mean": m, "std": s, "n": n})

        for mode in MODES:
            sub = sub_ds[sub_ds["supervision_type"] == mode]
            col = trained_col(mode)
            if sub.empty or col not in sub.columns:
                rows.append({"mode": mode, "dataset": ds, "mean": np.nan, "std": np.nan, "n": 0})
                continue
            m, s, n = per_seed_apply(sub,
                lambda g, _c=col: cf_auroc(g, _c, 0.85, higher_is_fail=True))
            rows.append({"mode": mode, "dataset": ds, "mean": m, "std": s, "n": n})

    df = pd.DataFrame(rows)

    mode_order = ["MSP", "scalar_benefit",
                  "gt_disagree", "gt_risk",
                  "dense_disagree", "dense_gap", "dense_multi"]

    nice_labels = {
        "MSP": "MSP",
        "scalar_benefit": "Scalar",
        "gt_disagree": "GT-Dis",
        "gt_risk": "GT-Gap",
        "dense_disagree": "T-Dis",
        "dense_gap": "T-Gap",
        "dense_multi": "T-Multi",
    }

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    xs = np.arange(len(mode_order))
    n_ds = len(datasets)
    bar_w = 0.78 / max(n_ds, 1)

    # 1) Background category bands first (so bars sit on top).
    y_lo, y_hi = 0.45, 1.0
    for (xmin, xmax, color, _label) in CATEGORY_BANDS:
        ax.axvspan(xmin, xmax, ymin=0, ymax=1, facecolor=color,
                   edgecolor="none", zorder=0, alpha=0.7)

    # 2) Bars: distinct color per dataset, consistent across all groups.
    for i, ds in enumerate(datasets):
        offset = (i - (n_ds - 1) / 2) * bar_w
        means, stds = [], []
        for m in mode_order:
            sub = df[(df["mode"] == m) & (df["dataset"] == ds)]
            if sub.empty:
                means.append(np.nan); stds.append(np.nan); continue
            mv = float(sub["mean"].iloc[0])
            sv = float(sub["std"].iloc[0])
            means.append(mv if np.isfinite(mv) else np.nan)
            stds.append(sv if np.isfinite(sv) else 0.0)
        means = np.asarray(means); stds = np.asarray(stds)

        color = DATASET_COLORS.get(ds, "#444444")
        ax.bar(xs + offset, np.nan_to_num(means, nan=0), width=bar_w,
               color=color, edgecolor="white", linewidth=0.6,
               label=DATASET_LABELS.get(ds, ds), zorder=3)
        # Error bars only where multi-seed.
        for xi, mv, sv in zip(xs, means, stds):
            if np.isfinite(mv) and sv > 0:
                ax.errorbar(xi + offset, mv, yerr=sv, fmt="none",
                            ecolor="#222", elinewidth=0.8, capsize=1.6,
                            capthick=0.8, zorder=4)

        # Value labels — placed above the upper cap so error bars never punch
        # through them.
        for xi, v, sv in zip(xs, means, stds):
            if np.isfinite(v) and v >= y_lo:
                ax.text(xi + offset, v + (sv if sv > 0 else 0) + 0.010,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                        rotation=90, color="#222", zorder=5)

    # 3) Vertical separators between supervision categories (subtle).
    for x in (0.5, 1.5, 3.5):
        ax.axvline(x, color="#bbb", lw=0.8, ls="--", zorder=1)

    # 4) Reference line at chance. Label outside the data area on the right
    # so no bar can cover it.
    ax.axhline(0.5, color="#888", ls=":", lw=0.9, zorder=2)
    ax.annotate("chance", xy=(1.001, 0.5), xycoords=("axes fraction", "data"),
                xytext=(4, 0), textcoords="offset points",
                fontsize=8, color="#666", va="center", ha="left",
                style="italic")

    # 5) X axis: mode labels.
    ax.set_xticks(xs)
    ax.set_xticklabels([nice_labels[m] for m in mode_order], fontsize=10)
    ax.set_xlim(-0.5, len(mode_order) - 0.5)

    # 6) Y axis.
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Confident-failure AUROC  (MSP ≥ 0.85, mit-b1)")
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    # 7) Category labels: place ABOVE the plot, below the legend, so nothing
    # collides with bars. We carve out vertical room via subplots_adjust.
    for (xmin, xmax, _color, label) in CATEGORY_BANDS:
        cx = 0.5 * (xmin + xmax)
        ax.text(cx, 1.015, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, color="#444",
                style="italic", linespacing=1.05)

    # 8) Legend ABOVE everything, horizontal, outside data area.
    legend_handles = [
        Patch(facecolor=DATASET_COLORS[ds], edgecolor="white",
              label=DATASET_LABELS.get(ds, ds))
        for ds in datasets
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=len(datasets),
               frameon=False, fontsize=9.5, handlelength=1.4,
               columnspacing=2.0, handletextpad=0.6)

    plt.subplots_adjust(left=0.07, right=0.955, bottom=0.13, top=0.83)
    savefig(fig, "fig_supervision_ablation")


if __name__ == "__main__":
    main()
