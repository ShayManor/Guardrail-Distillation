"""F3 — AURC horizontal bar chart (lower = better).

One subplot per dataset. Bars grouped by backbone, colored by method.
Dense-gap (ours) is highlighted. Oracle floor shown as faint dashed line.
Dynamically includes all datasets with data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, compute_aurc_table, mean_std_across_seeds,
    available_datasets, available_backbones,
    METHOD_COLORS, METHOD_LABELS, DATASET_LABELS,
)

METHODS = ("msp", "temp_msp", "mc_dropout", "dense_gap")  # oracle shown as floor line


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["supervision_type"] == "dense_multi"].copy()
    tbl = compute_aurc_table(pi)

    datasets = available_datasets(tbl)
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds / 2 + 3, 4.6), sharex=True,
                             squeeze=False)
    axes = axes[0]
    n_methods = len(METHODS)

    for ax, ds in zip(axes, datasets):
        sub = tbl[tbl["dataset"] == ds]
        backbones = available_backbones(sub)
        n_bb = len(backbones)
        bar_h = 0.16
        group_h = bar_h * (n_methods + 0.8)
        y_centers = np.arange(n_bb) * group_h

        for mi, m in enumerate(METHODS):
            vals = []
            errs = []
            for bb in backbones:
                s = sub[(sub["backbone"] == bb) & (sub["method"] == m)]
                agg = mean_std_across_seeds(s, ["backbone"], "aurc")
                if agg.empty or agg["mean"].isna().all():
                    vals.append(np.nan); errs.append(0)
                else:
                    vals.append(float(agg["mean"].iloc[0]))
                    std_v = agg["std"].iloc[0]
                    errs.append(float(std_v) if pd.notna(std_v) else 0.0)
            vals = np.array(vals, dtype=float)
            errs = np.array(errs, dtype=float)
            offset = (mi - (n_methods - 1) / 2) * bar_h
            color = METHOD_COLORS[m]
            edge = "#1a1a1a" if m == "dense_gap" else "none"
            ax.barh(y_centers + offset, vals, height=bar_h * 0.95,
                    color=color, edgecolor=edge, linewidth=1.0,
                    label=METHOD_LABELS[m],
                    xerr=errs if errs.any() else None,
                    error_kw={"ecolor": "#333333", "elinewidth": 0.8, "capsize": 2})

        # Oracle floor markers
        for bi, bb in enumerate(backbones):
            os_ = sub[(sub["backbone"] == bb) & (sub["method"] == "oracle")]
            if not os_.empty:
                val = os_["aurc"].mean()
                ax.axvline(val, ymin=(bi * group_h - group_h/2 + 0.02) / (n_bb * group_h),
                           ymax=((bi + 1) * group_h - group_h/2 - 0.02) / (n_bb * group_h),
                           color="#555", ls=":", lw=0.8, alpha=0.6)

        ax.set_yticks(y_centers, [b.replace("b", "mit-b") for b in backbones])
        ax.set_xlabel("AURC (lower is better)")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.grid(axis="y", alpha=0.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               columnspacing=2.0, handlelength=1.8)
    plt.subplots_adjust(bottom=0.22, top=0.90, wspace=0.18)

    savefig(fig, "fig_aurc_comparison")


if __name__ == "__main__":
    main()
