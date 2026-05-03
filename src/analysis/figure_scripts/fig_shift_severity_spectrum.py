"""F18 — Shift severity spectrum: how the dense head's advantage tracks difficulty.

X-axis: student mIoU (proxy for shift severity, lower = harder).
Y-axis: CF-AUROC delta (dense-gap minus MSP).

Each point is one (dataset, backbone) cell. Cityscapes cells cluster at the
easy end (high mIoU, delta ~0), ACDC cells at the hard end (low mIoU, large
delta), BDD in between. This visualizes the key claim: the guardrail's value
scales with how badly the student is broken.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, cf_auroc, available_datasets,
    available_backbones, pool_acdc_domain,
    BACKBONE_COLORS, BACKBONE_LABELS, DATASET_LABELS,
)

DATASET_MARKERS = {"city": "s", "acdc": "o", "idd": "^", "bdd": "D"}
DATASET_FACE_COLORS = {"city": "#cccccc", "acdc": "#4C72B0", "idd": "#55A868", "bdd": "#DD8452"}


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["supervision_type"] == "dense_multi"].copy()
    pi = pool_acdc_domain(pi)

    datasets = available_datasets(pi)
    points = []

    for ds in datasets:
        for bb in available_backbones(pi[pi["dataset"] == ds]):
            sub = pi[(pi["dataset"] == ds) & (pi["backbone"] == bb)]
            seed0 = sub["seed"].min()
            sub_s0 = sub[sub["seed"] == seed0]

            miou = float(sub_s0["student_miou"].mean())

            # Compute CF-AUROC for MSP and dense-gap (averaged over seeds)
            aurocs_msp, aurocs_dg = [], []
            for seed, s in sub.groupby("seed"):
                a_msp = cf_auroc(s, "student_msp", 0.85, higher_is_fail=False)
                a_dg = cf_auroc(s, "guardrailpp_utility_dense_gap", 0.85, higher_is_fail=True)
                if np.isfinite(a_msp):
                    aurocs_msp.append(a_msp)
                if np.isfinite(a_dg):
                    aurocs_dg.append(a_dg)

            if not aurocs_msp or not aurocs_dg:
                continue

            delta = np.mean(aurocs_dg) - np.mean(aurocs_msp)
            points.append({
                "dataset": ds, "backbone": bb,
                "miou": miou, "delta": delta,
                "dg_auroc": np.mean(aurocs_dg),
                "msp_auroc": np.mean(aurocs_msp),
            })

    df = pd.DataFrame(points)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        for _, row in sub.iterrows():
            bb = row["backbone"]
            ax.scatter(row["miou"], row["delta"],
                       s=140, marker=DATASET_MARKERS.get(ds, "o"),
                       facecolors=DATASET_FACE_COLORS.get(ds, "#999"),
                       edgecolors=BACKBONE_COLORS.get(bb, "#333"),
                       linewidths=2.0, zorder=5)
            ax.annotate(BACKBONE_LABELS[bb],
                        (row["miou"], row["delta"]),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=7.5, color="#333")

    # Trend line across all points
    if len(df) >= 3:
        m, b = np.polyfit(df["miou"], df["delta"], 1)
        xs = np.linspace(df["miou"].min() - 0.02, df["miou"].max() + 0.02, 50)
        ax.plot(xs, m * xs + b, color="#c0392b", lw=1.5, ls="--", alpha=0.6,
                label=f"trend: slope={m:.2f}")

    ax.axhline(0, color="#333", lw=0.8, ls=":")
    ax.set_xlabel("Student mIoU (higher = easier shift)")
    ax.set_ylabel("CF-AUROC delta (dense-gap minus MSP)")

    # Legend for datasets
    from matplotlib.lines import Line2D
    handles = []
    for ds in datasets:
        handles.append(Line2D([0], [0], marker=DATASET_MARKERS.get(ds, "o"),
                              color="none", markerfacecolor=DATASET_FACE_COLORS.get(ds, "#999"),
                              markeredgecolor="#333", markersize=10,
                              label=DATASET_LABELS.get(ds, ds)))
    if len(df) >= 3:
        handles.append(Line2D([0], [0], color="#c0392b", lw=1.5, ls="--", alpha=0.6,
                              label=f"trend (slope={m:.2f})"))
    ax.legend(handles=handles, fontsize=8.5, loc="upper right", frameon=False)

    ax.set_title("Guardrail value tracks shift severity across benchmarks\n"
                 "Harder shift (lower mIoU) = larger advantage over MSP",
                 fontsize=11)
    plt.subplots_adjust(top=0.86, bottom=0.13)
    savefig(fig, "fig_shift_severity_spectrum")


if __name__ == "__main__":
    main()
