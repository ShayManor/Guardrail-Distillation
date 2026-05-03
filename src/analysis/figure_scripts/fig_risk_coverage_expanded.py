"""Risk-coverage curves for ACDC and BDD.

Two panels: ACDC (left), BDD (right). Shows risk (y) vs coverage (x) for
each scoring method. Lower curve = better selective prediction.
Also prints AUGRC and risk@80/90/95% values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, pool_acdc_domain,
    METHOD_COLORS, METHOD_LABELS, DATASET_LABELS,
)

METHODS = [
    ("dense_gap", "guardrailpp_utility_dense_gap", True, "-", 2.4),
    ("msp",       "student_msp",                   False, "--", 1.6),
    ("temp_msp",  "temp_msp",                      False, ":", 1.2),
    ("mc_dropout", "mc_entropy",                    True, "-.", 1.2),
]


def risk_coverage(df, score_col, higher_is_fail, n_points=200):
    s = df.dropna(subset=[score_col, "student_risk"])
    scores = pd.to_numeric(s[score_col], errors="coerce").to_numpy()
    risks = pd.to_numeric(s["student_risk"], errors="coerce").to_numpy()
    mask = np.isfinite(scores) & np.isfinite(risks)
    scores, risks = scores[mask], risks[mask]
    if len(scores) < 20:
        return np.array([]), np.array([])
    if higher_is_fail:
        order = np.argsort(scores)
    else:
        order = np.argsort(-scores)
    sorted_risks = risks[order]
    n = len(sorted_risks)
    cum_risk = np.cumsum(sorted_risks) / (np.arange(n) + 1)
    covs = np.linspace(1.0 / n, 1.0, n_points)
    idxs = np.clip((covs * n).astype(int) - 1, 0, n - 1)
    return covs, cum_risk[idxs]


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[(pi["backbone"] == "b1") & (pi["supervision_type"] == "dense_multi")].copy()
    pi = pool_acdc_domain(pi)

    datasets = ["acdc", "bdd"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=False)

    for ax, ds in zip(axes, datasets):
        sub = pi[pi["dataset"] == ds]

        for method, col, hi, ls, lw in METHODS:
            covs, risks = risk_coverage(sub, col, hi)
            if len(covs) == 0:
                continue
            color = METHOD_COLORS[method]
            label = METHOD_LABELS[method]
            ax.plot(covs, risks, color=color, lw=lw, ls=ls, label=label)

        # Reference lines at 80/90/95% coverage
        for cov_ref in [0.80, 0.90, 0.95]:
            ax.axvline(cov_ref, color="#ccc", ls=":", lw=0.7, zorder=0)

        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selective risk")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlim(0.0, 1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=len(METHODS), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               columnspacing=2.0, handlelength=2.2, fontsize=9)
    plt.subplots_adjust(bottom=0.24, top=0.90, wspace=0.25, left=0.09,
                        right=0.97)
    savefig(fig, "fig_risk_coverage_expanded")


if __name__ == "__main__":
    main()
