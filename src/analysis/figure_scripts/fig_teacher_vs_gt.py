"""Central figure: Distill-supervised vs GT-supervised guardrail under
distribution shift.

Left panel: CF-AUROC across the three OOD datasets (ACDC, IDD, BDD100K),
grouped by dataset. Cityscapes (in-domain) is excluded — in-distribution
MaxLogit dominates via the well-known softmax-saturation effect and is
covered separately in the supplementary calibration analysis.

Right panel: Distill-vs-GT delta across datasets — the distillation
advantage emerges under domain shift and grows with shift severity.

This is the paper's key result figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from _lib import (
    apply_style, load_table, savefig, cf_auroc, cf_auroc_stratified,
    pool_acdc_domain, available_datasets, DATASET_LABELS,
)

# ACDC uses a stratified failure cutoff (per fog/night/rain/snow) at a
# higher MSP floor so a single condition's base rate can't own the label.
# Other datasets stay on the pooled cutoff at the original threshold.
ACDC_MSP_THR = 0.85
DEFAULT_MSP_THR = 0.85
# Supervision modes → trained column
MODE_COL = {
    "dense_multi": "guardrailpp_utility_dense_gap",
    "gt_disagree": "guardrailpp_utility_dense_bce",
    "gt_risk":     "guardrailpp_utility_dense_gap",
}

# Datasets shown in the figure (OOD only; Cityscapes excluded — see
# module docstring).
OOD_DATASETS = ("acdc", "idd", "bdd")

# Post-hoc baselines: (label, column, higher_is_fail)
POSTHOC = [
    ("MSP",        "student_msp",     False),
    ("Entropy",    "student_entropy", True),
    ("MaxLogit",   "max_logit",       False),
    ("MC-Dropout", "mc_entropy",      True),
]

# Display order
ALL_METHODS = [
    "MSP", "Entropy", "MaxLogit", "MC-Dropout",
    "GT-Dis", "GT-Gap",
    "T-Multi (ours)",
]

COLORS = {
    "MSP":            "#b0b0b0",
    "Entropy":        "#DD8452",
    "MaxLogit":       "#64B5CD",
    "MC-Dropout":     "#C44E52",
    "GT-Dis":         "#55A868",
    "GT-Gap":         "#CCB974",
    "T-Multi (ours)": "#7B68AD",
}

SHORT_DS = {
    "acdc": "ACDC\n(weather)",
    "idd":  "IDD\n(geo+semantic)",
    "bdd":  "BDD100K\n(geographic)",
}


def _score_auroc(sub, col, hi, ds):
    """Route ACDC rows through the condition-stratified AUROC with the ACDC
    threshold; other datasets use the standard pooled AUROC at the default
    threshold. The ACDC pool mixes fog/night/rain/snow with very different
    logit scales and base failure rates, so the pooled top-20% cutoff puts
    18 of 19 confident failures in night at MSP>=0.85 — collapsing the task
    into a night classifier. Per-condition cutoffs restore a fair label."""
    if ds == "acdc":
        return cf_auroc_stratified(sub, col, ACDC_MSP_THR,
                                   group_col="condition",
                                   higher_is_fail=hi)
    return cf_auroc(sub, col, DEFAULT_MSP_THR, higher_is_fail=hi)


def compute_aurocs(pi, datasets):
    """Compute CF-AUROC for every (method, dataset) pair."""
    rows = []
    for ds in datasets:
        ds_data = pi[pi["dataset"] == ds]
        # Post-hoc scores are identical across supervision types
        sub_any = ds_data[ds_data["supervision_type"] == "dense_multi"]
        if sub_any.empty:
            sub_any = ds_data.drop_duplicates(subset=["image_id"])

        for label, col, hi in POSTHOC:
            if col not in sub_any.columns:
                rows.append({"method": label, "dataset": ds, "auroc": np.nan})
                continue
            a = _score_auroc(sub_any, col, hi, ds)
            rows.append({"method": label, "dataset": ds, "auroc": a})

        mode_labels = {
            "dense_multi": "T-Multi (ours)",
            "gt_disagree": "GT-Dis",
            "gt_risk":     "GT-Gap",
        }
        for mode, col in MODE_COL.items():
            sub = ds_data[ds_data["supervision_type"] == mode]
            label = mode_labels[mode]
            if sub.empty or col not in sub.columns:
                rows.append({"method": label, "dataset": ds, "auroc": np.nan})
                continue
            a = _score_auroc(sub, col, True, ds)
            rows.append({"method": label, "dataset": ds, "auroc": a})

    return pd.DataFrame(rows)


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pool_acdc_domain(pi)
    pi = pi[pi["backbone"] == "b1"].copy()

    present = set(available_datasets(pi))
    datasets = [ds for ds in OOD_DATASETS if ds in present]
    df = compute_aurocs(pi, datasets)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4),
                             gridspec_kw={"width_ratios": [2.2, 1]})

    # ── Left panel: grouped bars ──
    ax = axes[0]
    n_methods = len(ALL_METHODS)
    n_ds = len(datasets)
    bar_w = 0.78 / n_methods
    xs = np.arange(n_ds)

    for j, method in enumerate(ALL_METHODS):
        offset = (j - (n_methods - 1) / 2) * bar_w
        vals = []
        for ds in datasets:
            sub = df[(df["method"] == method) & (df["dataset"] == ds)]
            v = float(sub["auroc"].iloc[0]) if not sub.empty and np.isfinite(sub["auroc"].iloc[0]) else np.nan
            vals.append(v)
        vals_arr = np.array(vals)
        is_ours = "ours" in method
        ax.bar(xs + offset, np.nan_to_num(vals_arr, nan=0), bar_w,
               color=COLORS[method],
               edgecolor="#1a1a1a" if is_ours else "none",
               linewidth=1.3 if is_ours else 0,
               label=method, zorder=3 if is_ours else 2)
        for i, v in enumerate(vals):
            if np.isfinite(v) and v > 0.5:
                ax.text(xs[i] + offset, v + 0.006, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5.8, rotation=90)

    # Banner above the bars (replaces the in-domain / OOD italics header).
    ax.text(0.5, 1.005, "distribution shift (OOD)",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color="#888", style="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT_DS.get(ds, ds) for ds in datasets], fontsize=9)
    ax.set_ylim(0.48, 0.80)
    ax.set_ylabel("Confident-failure AUROC (MSP ≥ 0.85)")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.4)
    # Legend lifted out of the data area entirely.
    ax.legend(fontsize=8.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.07), ncol=7, frameon=False,
              handlelength=1.2, columnspacing=1.4, handletextpad=0.5)

    # ── Right panel: T-Multi − GT delta ──
    ax2 = axes[1]
    delta_datasets = datasets

    gt_methods = ["GT-Dis", "GT-Gap"]
    gt_colors = [COLORS["GT-Dis"], COLORS["GT-Gap"]]
    xs2 = np.arange(len(delta_datasets))
    bw = 0.32

    for k, (gt_m, gc) in enumerate(zip(gt_methods, gt_colors)):
        deltas = []
        for ds in delta_datasets:
            t_val = df[(df["method"] == "T-Multi (ours)") & (df["dataset"] == ds)]["auroc"]
            g_val = df[(df["method"] == gt_m) & (df["dataset"] == ds)]["auroc"]
            t = float(t_val.iloc[0]) if not t_val.empty else np.nan
            g = float(g_val.iloc[0]) if not g_val.empty else np.nan
            deltas.append(t - g if np.isfinite(t) and np.isfinite(g) else np.nan)
        deltas = np.array(deltas)
        offset = (k - 0.5) * bw
        bars = ax2.bar(xs2 + offset, deltas * 100, bw,
                       color=gc, edgecolor="#1a1a1a", linewidth=0.8,
                       alpha=0.85, label=f"vs {gt_m}")
        for bar, d in zip(bars, deltas):
            if np.isfinite(d):
                y = d * 100
                va = "bottom" if y >= 0 else "top"
                off = 0.18 if y >= 0 else -0.18
                color = "#1b7a28" if d > 0.002 else "#a6201a" if d < -0.002 else "#666"
                ax2.text(bar.get_x() + bar.get_width() / 2, y + off,
                         f"{d*100:+.1f}", ha="center", va=va,
                         fontsize=7.5, fontweight="bold", color=color)

    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels([SHORT_DS.get(ds, ds).split("\n")[0] for ds in delta_datasets],
                        fontsize=9)
    ax2.set_ylabel("T-Multi advantage over GT (pp)")
    # Generous y-limits so neither +4.2 nor -0.5 crowds an edge.
    ax2.set_ylim(-0.8, 5.8)
    # Legend lifted above the panel; matches left-panel style.
    ax2.legend(fontsize=8.5, loc="lower center",
               bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False,
               handlelength=1.2, columnspacing=1.4, handletextpad=0.5)

    plt.subplots_adjust(wspace=0.28, top=0.85, bottom=0.16, left=0.06, right=0.97)
    savefig(fig, "fig_teacher_vs_gt")


if __name__ == "__main__":
    main()
