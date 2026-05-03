"""F5 — Teacher-routing Pareto curves.

Rows: datasets (dynamically includes all available). Columns: metrics
(effective_risk, benefit_recovered_frac). 3 backbones per axis. Solid =
dense-gap guardrail, dashed = MSP, dotted grey = oracle and random.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _lib import (
    apply_style, load_table, savefig, mean_std_across_seeds,
    available_datasets, available_backbones, pool_acdc_domain,
    BACKBONE_COLORS, BACKBONE_LABELS, DATASET_LABELS,
)


def main():
    apply_style()
    tb = load_table("teacher_budget")
    tb = tb[tb["supervision_type"] == "dense_multi"].copy()
    tb = pool_acdc_domain(tb)

    metrics = [
        ("effective_risk",         "Effective risk (lower is better)"),
        ("benefit_recovered_frac", "Oracle headroom recovered (higher is better)"),
    ]
    datasets = available_datasets(tb)
    n_ds = len(datasets)

    fig, axes = plt.subplots(n_ds, 2, figsize=(11.5, 3.4 * n_ds), sharex=True,
                             squeeze=False)

    for row_i, ds in enumerate(datasets):
        for col_i, (ycol, ylabel) in enumerate(metrics):
            ax = axes[row_i, col_i]
            sub_ds = tb[tb["dataset"] == ds]
            for bb in available_backbones(sub_ds):
                s_bb = sub_ds[sub_ds["backbone"] == bb]
                if s_bb.empty:
                    continue
                color = BACKBONE_COLORS[bb]

                for m, ls, lw, marker in (
                    ("guardrailpp_utility", "-",  2.3, "o"),
                    ("msp",                 "--", 1.3, ""),
                ):
                    s = s_bb[s_bb["method"] == m]
                    if s.empty:
                        continue
                    agg = mean_std_across_seeds(s, ["teacher_budget"], ycol).sort_values("teacher_budget")
                    ax.plot(agg["teacher_budget"], agg["mean"], color=color,
                            lw=lw, linestyle=ls, marker=marker, markersize=4,
                            alpha=0.95 if ls == "-" else 0.8)
                    if agg["std"].notna().any() and (agg["std"] > 0).any():
                        ax.fill_between(agg["teacher_budget"],
                                        agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                                        color=color, alpha=0.15, linewidth=0)

                for m, alpha in (("oracle", 0.6), ("random", 0.45)):
                    s = s_bb[s_bb["method"] == m]
                    if s.empty:
                        continue
                    agg = mean_std_across_seeds(s, ["teacher_budget"], ycol).sort_values("teacher_budget")
                    ax.plot(agg["teacher_budget"], agg["mean"],
                            color="#777" if m == "oracle" else "#bbb",
                            lw=1.0, linestyle=":", alpha=alpha)

            ax.set_xlim(0, 1)
            if row_i == n_ds - 1:
                ax.set_xlabel("Teacher budget (fraction routed)")
            ds_label = DATASET_LABELS.get(ds, ds)
            if col_i == 0:
                ax.set_ylabel(f"{ds_label}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)

    legend_handles = []
    legend_labels = []
    for bb in ("b0", "b1", "b2"):
        legend_handles.append(Line2D([0], [0], color=BACKBONE_COLORS[bb], lw=2.3, marker="o", markersize=5))
        legend_labels.append(f"{BACKBONE_LABELS[bb]} dense-gap")
    for bb in ("b0", "b1", "b2"):
        legend_handles.append(Line2D([0], [0], color=BACKBONE_COLORS[bb], lw=1.3, linestyle="--"))
        legend_labels.append(f"{BACKBONE_LABELS[bb]} MSP")
    legend_handles += [
        Line2D([0], [0], color="#777", lw=1.0, linestyle=":"),
        Line2D([0], [0], color="#bbb", lw=0.9, linestyle=":"),
    ]
    legend_labels += ["oracle", "random"]

    fig.legend(legend_handles, legend_labels, ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9,
               columnspacing=1.6, handlelength=2.2)
    plt.subplots_adjust(bottom=0.08, top=0.95, wspace=0.22, hspace=0.15)

    savefig(fig, "fig_teacher_budget")


if __name__ == "__main__":
    main()
