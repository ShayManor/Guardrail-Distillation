"""Risk-coverage curves: Teacher vs GT vs post-hoc baselines.

One subplot per dataset (b1 only). Shows risk-coverage for:
  - Teacher guardrail (dense_multi, solid thick purple)
  - GT guardrail (gt_disagree, solid green)
  - MaxLogit (dashed blue)
  - MSP (dashed grey)
  - Oracle (dotted grey, unreachable floor)

Lower risk at any given coverage is better.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _lib import (
    apply_style, load_table, savefig, risk_coverage_curve,
    available_datasets, pool_acdc_domain, DATASET_LABELS, per_seed_curve,
)

# Methods to plot: (label, supervision_type, column, higher_is_fail, color, ls, lw, alpha)
METHODS = [
    ("MSP",             "dense_multi", "student_msp",                   False, "#aaaaaa", "--", 1.4, 0.9),
    ("MaxLogit",        "dense_multi", "max_logit",                     False, "#64B5CD", "--", 1.6, 0.9),
    ("MC-Dropout",      "dense_multi", "mc_entropy",                    True,  "#C44E52", "--", 1.2, 0.7),
    ("GT-Dis",          "gt_disagree", "guardrailpp_utility_dense_bce", True,  "#55A868", "-",  2.0, 0.9),
    ("GT-Gap",          "gt_risk",     "guardrailpp_utility_dense_gap", True,  "#CCB974", "-",  1.8, 0.85),
    ("T-Multi (ours)",  "dense_multi", "guardrailpp_utility_dense_gap", True,  "#7B68AD", "-",  2.5, 1.0),
]


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["backbone"] == "b1"].copy()
    pi = pool_acdc_domain(pi)

    datasets = available_datasets(pi)
    n_ds = len(datasets)
    ncols = 2
    nrows = (n_ds + 1) // 2
    fig, axes_2d = plt.subplots(nrows, ncols, figsize=(10, 8), sharey=False)
    axes = axes_2d.flatten()[:n_ds]

    cov_grid = np.linspace(0.05, 1.0, 60)

    for ax, ds in zip(axes, datasets):
        ds_data = pi[pi["dataset"] == ds]

        for label, stype, col, hi, color, ls, lw, alpha in METHODS:
            sub = ds_data[ds_data["supervision_type"] == stype]
            if sub.empty or col not in sub.columns:
                continue
            mean, std, n = per_seed_curve(sub,
                lambda g, _c=col, _h=hi: risk_coverage_curve(g, _c, _h, n_points=50),
                cov_grid)
            if not np.isfinite(mean).any():
                continue
            ax.plot(cov_grid, mean, color=color, lw=lw, ls=ls, alpha=alpha, label=label)
            if n >= 2 and np.isfinite(std).any():
                ax.fill_between(cov_grid, mean - std, mean + std,
                                color=color, alpha=alpha * 0.20, lw=0)

        # Oracle floor
        sub_any = ds_data[ds_data["supervision_type"] == "dense_multi"]
        if not sub_any.empty:
            mean, std, n = per_seed_curve(sub_any,
                lambda g: risk_coverage_curve(g, "student_risk", True, n_points=50),
                cov_grid)
            if np.isfinite(mean).any():
                ax.plot(cov_grid, mean, color="#888", lw=1.0, ls=":", alpha=0.5, label="Oracle")

        ax.set_xlim(0.05, 1.0)
        ax.set_xlabel("Coverage")
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=10)
        ax.set_ylim(0, None)

    for ax in axes:
        ax.set_ylabel("Selective risk (lower is better)")

    # Shared legend
    handles = []
    for label, _, _, _, color, ls, lw, alpha in METHODS:
        handles.append(Line2D([0], [0], color=color, lw=lw, ls=ls, alpha=alpha, label=label))
    handles.append(Line2D([0], [0], color="#888", lw=1.0, ls=":", alpha=0.5, label="Oracle"))

    fig.legend(handles=handles, ncols=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               frameon=False, columnspacing=1.2, handlelength=2.2, fontsize=8)
    # Hide unused axes
    for i in range(n_ds, len(axes_2d.flatten())):
        axes_2d.flatten()[i].set_visible(False)

    plt.subplots_adjust(bottom=0.10, top=0.95, wspace=0.28, hspace=0.32,
                        left=0.07, right=0.97)
    savefig(fig, "fig_risk_coverage")


if __name__ == "__main__":
    main()
