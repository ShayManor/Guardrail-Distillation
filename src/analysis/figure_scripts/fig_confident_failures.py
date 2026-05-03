"""F1 — Confident-failure AUROC vs MSP threshold.

One subplot per dataset (CS-val, ACDC, IDD, BDD100K). One line per backbone using
the trained dense-gap head, plus a dashed MSP baseline per backbone (same
color, thinner). Multi-seed = shaded mean +/- std band; single-seed = line only.

Data source: ``combined_all/per_image.csv`` — AUROC is computed live using the
CW-TR top-20% failure definition, so the figure is immune to any legacy
`guardrail_auroc` column corruption on old b0/b2 eval runs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, compute_cf_auroc_table,
    mean_std_across_seeds, available_datasets,
    BACKBONE_COLORS, BACKBONE_LABELS, DATASET_LABELS,
)


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["supervision_type"] == "dense_multi"].copy()
    cf = compute_cf_auroc_table(pi, msp_thresholds=(0.85, 0.88, 0.90, 0.92, 0.95))

    datasets = available_datasets(cf)
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 4.4), sharey=True,
                             squeeze=False)
    axes = axes[0]

    legend_h, legend_l = [], []
    for ax, ds in zip(axes, datasets):
        sub = cf[cf["dataset"] == ds]
        if sub.empty:
            ax.set_visible(False)
            continue
        for bb in ("b0", "b1", "b2"):
            s = sub[sub["backbone"] == bb]
            if s.empty:
                continue
            color = BACKBONE_COLORS[bb]
            # Dense-gap trained head
            dg = s[s["method"] == "dense_gap"].sort_values("msp_threshold")
            agg = mean_std_across_seeds(dg, ["msp_threshold"], "auroc")
            l1, = ax.plot(agg["msp_threshold"], agg["mean"], color=color, lw=2.4,
                          marker="o", markersize=5,
                          label=f"{BACKBONE_LABELS[bb]} dense-gap")
            if agg["std"].notna().any() and (agg["std"] > 0).any():
                ax.fill_between(agg["msp_threshold"],
                                agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                                color=color, alpha=0.18, linewidth=0)
            # MSP baseline
            msp = s[s["method"] == "msp"].sort_values("msp_threshold")
            agg_msp = mean_std_across_seeds(msp, ["msp_threshold"], "auroc")
            l2, = ax.plot(agg_msp["msp_threshold"], agg_msp["mean"], color=color,
                          lw=1.3, linestyle="--", alpha=0.85,
                          label=f"{BACKBONE_LABELS[bb]} MSP")
            if ax is axes[0]:
                legend_h += [l1, l2]
                legend_l += [f"{BACKBONE_LABELS[bb]} dense-gap", f"{BACKBONE_LABELS[bb]} MSP"]

        ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.5)
        ax.set_xlabel("MSP confidence threshold")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlim(0.84, 0.96)
        ax.set_ylim(0.40, 1.03)

    axes[0].set_ylabel("Confident-failure AUROC")

    fig.legend(legend_h, legend_l, ncols=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               columnspacing=1.6, handlelength=2.2)
    plt.subplots_adjust(bottom=0.23, top=0.90, wspace=0.08)
    savefig(fig, "fig_confident_failures")


if __name__ == "__main__":
    main()
