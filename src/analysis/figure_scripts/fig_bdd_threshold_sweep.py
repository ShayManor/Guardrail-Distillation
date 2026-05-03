"""Threshold sweep: CF-AUROC across MSP thresholds.

Shows how Teacher, GT, MaxLogit, and MSP degrade as the confidence bar rises.
One panel per OOD dataset + Cityscapes. The teacher guardrail degrades more
slowly than GT and post-hoc methods under strict thresholds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, cf_auroc, pool_acdc_domain,
    available_datasets, DATASET_LABELS,
)

# Which score to use for each supervision type
MODE_COL = {
    "dense_multi": ("guardrailpp_utility_dense_gap", True),
    "gt_disagree": ("guardrailpp_utility_dense_bce", True),
    "gt_risk":     ("guardrailpp_utility_dense_gap", True),
}

# Display config: (label, color, ls, lw, marker, ms)
METHOD_STYLE = {
    "T-Multi (ours)": ("#7B68AD", "-",  2.5, "o", 6),
    "GT-Dis":         ("#55A868", "-",  1.8, "D", 4),
    "GT-Gap":         ("#CCB974", "-",  1.6, "v", 4),
    "MaxLogit":       ("#64B5CD", "--", 1.8, "s", 4),
    "MSP":            ("#aaaaaa", "--", 1.4, "^", 4),
}

THRESHOLDS = (0.85, 0.88, 0.90, 0.92, 0.95, 0.97)


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["backbone"] == "b1"].copy()
    pi = pool_acdc_domain(pi)

    datasets = available_datasets(pi)
    n_ds = len(datasets)
    ncols = 2
    nrows = (n_ds + 1) // 2
    fig, axes_2d = plt.subplots(nrows, ncols, figsize=(10, 7.5), sharey=True)
    axes = axes_2d.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_data = pi[pi["dataset"] == ds]

        # Post-hoc methods (use dense_multi rows for student scores)
        sub_any = ds_data[ds_data["supervision_type"] == "dense_multi"]
        if sub_any.empty:
            sub_any = ds_data.drop_duplicates(subset=["image_id"])

        # MSP
        msp_vals = []
        for thr in THRESHOLDS:
            tmp = sub_any.copy()
            tmp["_neg_msp"] = -pd.to_numeric(tmp["student_msp"], errors="coerce")
            msp_vals.append(cf_auroc(tmp, "_neg_msp", thr, higher_is_fail=True))
        color, ls, lw, marker, ms = METHOD_STYLE["MSP"]
        ax.plot(THRESHOLDS, msp_vals, color=color, ls=ls, lw=lw, marker=marker,
                markersize=ms, label="MSP")

        # MaxLogit
        ml_vals = []
        for thr in THRESHOLDS:
            tmp = sub_any.copy()
            tmp["_neg_ml"] = -pd.to_numeric(tmp["max_logit"], errors="coerce")
            ml_vals.append(cf_auroc(tmp, "_neg_ml", thr, higher_is_fail=True))
        color, ls, lw, marker, ms = METHOD_STYLE["MaxLogit"]
        ax.plot(THRESHOLDS, ml_vals, color=color, ls=ls, lw=lw, marker=marker,
                markersize=ms, label="MaxLogit")

        # Learned heads
        mode_labels = {
            "gt_disagree": "GT-Dis",
            "gt_risk": "GT-Gap",
            "dense_multi": "T-Multi (ours)",
        }
        for mode in ("gt_disagree", "gt_risk", "dense_multi"):
            col, hi = MODE_COL[mode]
            sub = ds_data[ds_data["supervision_type"] == mode]
            if sub.empty or col not in sub.columns:
                continue
            vals = [cf_auroc(sub, col, thr, higher_is_fail=hi) for thr in THRESHOLDS]
            label = mode_labels[mode]
            color, ls, lw, marker, ms = METHOD_STYLE[label]
            ax.plot(THRESHOLDS, vals, color=color, ls=ls, lw=lw, marker=marker,
                    markersize=ms, label=label)

        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.4)
        ax.set_xlabel("MSP threshold")
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=10)
        ax.set_xlim(0.84, 0.975)
        ax.set_ylim(0.30, 1.02)
        ax.set_xticks(THRESHOLDS)
        ax.set_xticklabels([f"{t:.2f}" for t in THRESHOLDS], fontsize=7.5)

    axes[0].set_ylabel("CF-AUROC")
    axes[2].set_ylabel("CF-AUROC")

    # Hide unused axes
    for i in range(n_ds, len(axes)):
        axes[i].set_visible(False)

    # Shared legend from the first panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, -0.01), frameon=False,
               columnspacing=1.5, handlelength=2.2, fontsize=8.5)

    plt.subplots_adjust(bottom=0.12, top=0.94, wspace=0.08, hspace=0.32,
                        left=0.07, right=0.97)
    savefig(fig, "fig_threshold_sweep")


if __name__ == "__main__":
    main()
