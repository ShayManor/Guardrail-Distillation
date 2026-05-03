"""F10 — Variance decomposition of teacher_benefit.

Var(benefit) = Var(SR) + Var(TR) - 2*Cov(SR, TR)

For each backbone on each dataset, compute the decomposition. Stacked bar
chart shows how the two variances almost cancel out under the +0.8 correlation.
Dynamically includes all datasets with data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, available_datasets, available_backbones,
    pool_acdc_domain, BACKBONE_LABELS, DATASET_LABELS,
)


def variances(sub: pd.DataFrame):
    sr = pd.to_numeric(sub["student_risk"], errors="coerce")
    tr = pd.to_numeric(sub["teacher_risk"], errors="coerce")
    bn = pd.to_numeric(sub["teacher_benefit"], errors="coerce")
    mask = sr.notna() & tr.notna() & bn.notna()
    sr, tr, bn = sr[mask], tr[mask], bn[mask]
    if len(sr) < 3:
        return None
    var_sr = float(sr.var(ddof=1))
    var_tr = float(tr.var(ddof=1))
    cov = float(sr.cov(tr))
    var_bn = float(bn.var(ddof=1))
    predicted = var_sr + var_tr - 2 * cov
    r = float(sr.corr(tr))
    return {
        "var_sr": var_sr, "var_tr": var_tr,
        "two_cov": 2 * cov, "var_bn": var_bn,
        "predicted": predicted, "r": r, "n": len(sr),
    }


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["supervision_type"] == "dense_multi"].copy()
    pi = pool_acdc_domain(pi)

    rows = []
    datasets = available_datasets(pi)
    for ds in datasets:
        for bb in available_backbones(pi[pi["dataset"] == ds]):
            sub = pi[(pi["dataset"] == ds) & (pi["backbone"] == bb)]
            v = variances(sub)
            if v is None:
                continue
            rows.append({"dataset": ds, "backbone": bb, **v})
    df = pd.DataFrame(rows)

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds / 2 + 3, 4.6), sharey=False,
                             squeeze=False)
    axes = axes[0]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        bbs = [bb for bb in ("b0", "b1", "b2") if bb in sub["backbone"].values]
        xs = np.arange(len(bbs))
        sub_sorted = sub.set_index("backbone").reindex(bbs)

        pos = sub_sorted["var_sr"].values + sub_sorted["var_tr"].values
        neg = sub_sorted["two_cov"].values
        residual = sub_sorted["var_bn"].values

        ax.bar(xs - 0.17, sub_sorted["var_sr"].values, width=0.30,
               color="#4C72B0", label="Var(student_risk)")
        ax.bar(xs - 0.17, sub_sorted["var_tr"].values,
               bottom=sub_sorted["var_sr"].values,
               width=0.30, color="#55A868", label="Var(teacher_risk)")
        ax.bar(xs + 0.17, neg, width=0.30,
               color="#C44E52", alpha=0.85, label="2*Cov(SR, TR)  (subtracted)")
        ax.bar(xs + 0.17, residual, width=0.18,
               color="#8172B3",
               label="Var(teacher_benefit)")

        for xi, bb in enumerate(bbs):
            row = sub_sorted.loc[bb]
            if pd.isna(row.get("r")):
                continue
            ax.text(xi, max(pos[xi], neg[xi]) * 1.05,
                    f"r={row['r']:+.2f}",
                    ha="center", va="bottom", fontsize=9, color="#333")

        ax.set_xticks(xs, [BACKBONE_LABELS[b] for b in bbs])
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_ylabel("Variance (per-image units)")
        ax.axhline(0, color="#333", lw=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=8.5,
               columnspacing=1.8, handlelength=1.8)
    fig.suptitle("Var(benefit) = Var(SR) + Var(TR) - 2*Cov(SR, TR)\n"
                 "High correlation cancels almost everything", fontsize=11, y=1.02)
    plt.subplots_adjust(bottom=0.24, top=0.84, wspace=0.26)

    savefig(fig, "fig_negative_result_decomp")


if __name__ == "__main__":
    main()
