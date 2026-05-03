"""Within-condition failure detection on ACDC.

Addresses shortcut-learning critique: shows failure AUROC *within* each
ACDC condition (fog/night/rain/snow). Includes GT baseline to show
teacher supervision wins within-condition too.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from _lib import apply_style, load_table, savefig


# (label, supervision_type, column, higher_is_fail, color)
METHODS = [
    ("MSP",            "dense_multi", "student_msp",                   False, "#aaaaaa"),
    ("MaxLogit",       "dense_multi", "max_logit",                     False, "#64B5CD"),
    ("MC-Dropout",     "dense_multi", "mc_entropy",                    True,  "#C44E52"),
    ("GT-Dis",         "gt_disagree", "guardrailpp_utility_dense_bce", True,  "#55A868"),
    ("T-Multi (ours)", "dense_multi", "guardrailpp_utility_dense_gap", True,  "#7B68AD"),
]

CONDITIONS = ["fog", "night", "rain", "snow"]
COND_LABELS = {"fog": "Fog", "night": "Night", "rain": "Rain", "snow": "Snow"}


def failure_auroc(df, score_col, higher_is_fail, q=0.20):
    s = df.dropna(subset=[score_col, "student_risk"])
    scores = pd.to_numeric(s[score_col], errors="coerce").to_numpy()
    risks = pd.to_numeric(s["student_risk"], errors="coerce").to_numpy()
    mask = np.isfinite(scores) & np.isfinite(risks)
    scores, risks = scores[mask], risks[mask]
    if len(scores) < 20:
        return np.nan
    k = max(1, int(round(q * len(risks))))
    cutoff = np.sort(risks)[-k]
    y = (risks >= cutoff).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    s_final = scores if higher_is_fail else -scores
    return roc_auc_score(y, s_final)


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[(pi["backbone"] == "b1") & (pi["dataset"] == "acdc")].copy()

    n_methods = len(METHODS)
    n_conds = len(CONDITIONS)
    x = np.arange(n_conds)
    width = 0.82 / n_methods

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    for i, (label, stype, col, hi, color) in enumerate(METHODS):
        vals = []
        sub_type = pi[pi["supervision_type"] == stype]
        for cond in CONDITIONS:
            sub = sub_type[sub_type["domain"] == cond]
            if sub.empty or col not in sub.columns:
                vals.append(np.nan)
            else:
                vals.append(failure_auroc(sub, col, hi))
        offset = (i - (n_methods - 1) / 2) * width
        is_ours = "ours" in label
        bars = ax.bar(x + offset, vals, width=width * 0.9, color=color,
                      label=label,
                      edgecolor="#1a1a1a" if is_ours else "white",
                      linewidth=1.2 if is_ours else 0.5)
        # Value labels
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=10)
    ax.set_ylabel("Failure AUROC (within-condition)")
    ax.set_ylim(0.35, 0.90)

    ax.legend(ncols=3, loc="upper right", fontsize=7.5, frameon=False)
    plt.tight_layout()
    savefig(fig, "fig_within_condition")


if __name__ == "__main__":
    main()
