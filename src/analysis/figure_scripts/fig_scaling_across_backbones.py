"""F2 — Scaling: confident-failure AUROC @ msp>=0.85 across backbones.

One subplot per dataset. X axis = backbone (b0, b1, b2). Lines are different
detectors (dense-gap, MSP, temp_MSP, MC-dropout). Dynamically includes all
datasets with data (Cityscapes, ACDC, BDD100K).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, compute_cf_auroc_table,
    mean_std_across_seeds, available_datasets, available_backbones,
    METHOD_COLORS, METHOD_LABELS, DATASET_LABELS,
)

METHODS = ("dense_gap", "msp", "temp_msp", "mc_dropout")


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["supervision_type"] == "dense_multi"].copy()
    cf = compute_cf_auroc_table(pi, msp_thresholds=(0.85,))
    cf = cf[cf["msp_threshold"] == 0.85]

    datasets = available_datasets(cf)
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.0 * n_ds, 3.6), sharey=True,
                             squeeze=False)
    axes = axes[0]

    for ax, ds in zip(axes, datasets):
        sub = cf[cf["dataset"] == ds]
        backbones = available_backbones(sub)
        xs = np.arange(len(backbones))

        for m in METHODS:
            ms = sub[sub["method"] == m]
            agg = mean_std_across_seeds(ms, ["backbone"], "auroc")
            agg = agg.set_index("backbone").reindex(backbones).reset_index()
            ys = agg["mean"].values
            es = agg["std"].values
            color = METHOD_COLORS[m]
            lw = 2.6 if m == "dense_gap" else 1.4
            marker = "o" if m == "dense_gap" else "s"
            ms_size = 8 if m == "dense_gap" else 5
            ax.plot(xs, ys, color=color, marker=marker, lw=lw, markersize=ms_size,
                    label=METHOD_LABELS[m])
            has_std = np.isfinite(es).any() and (es > 0).any()
            if has_std:
                ax.errorbar(xs, ys, yerr=es, color=color, lw=0, elinewidth=1.3,
                            capsize=3)

        ax.set_xticks(xs, [b.replace("b", "mit-b") for b in backbones])
        ax.set_xlabel("Student backbone")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_ylim(0.40, 1.02)
        ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.5)

    axes[0].set_ylabel("Conf-fail AUROC @ msp >= 0.85")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               columnspacing=2.0, handlelength=2.2)
    plt.subplots_adjust(bottom=0.22, top=0.90, wspace=0.1)

    savefig(fig, "fig_scaling_across_backbones")


if __name__ == "__main__":
    main()
