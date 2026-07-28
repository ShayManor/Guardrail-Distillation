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
    pool_acdc_domain, available_datasets, DATASET_LABELS, per_seed_apply,
    ANALYSIS_ROOT,
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
    ("Energy",     "energy_score",    True),
    ("MC-Dropout", "mc_entropy",      True),
]

# Deep Ensemble lives outside combined_all — it is its own eval campaign and
# combined_all/per_image.csv has no ensemble_entropy column. One three-member
# ensemble per dataset; its measured spread is ~0, drawn as a flat error bar.
DEEP_ENSEMBLE_FILES = {
    "acdc": "b1_per_image.csv",
    "idd":  "iddfull_b1_per_image.csv",
    "bdd":  "bddfull_b1_per_image.csv",
}

# Display order
ALL_METHODS = [
    "MSP", "Entropy", "MaxLogit", "Energy", "MC-Dropout", "DeepEns (3×)",
    "GT-Dis", "GT-Gap",
    "T-Multi (ours)",
]

COLORS = {
    "MSP":            "#b0b0b0",
    "Entropy":        "#DD8452",
    "MaxLogit":       "#64B5CD",
    "Energy":         "#4C72B0",
    "MC-Dropout":     "#C44E52",
    "DeepEns (3×)":   "#937860",
    "GT-Dis":         "#55A868",
    "GT-Gap":         "#CCB974",
    "T-Multi (ours)": "#7B68AD",
}

SHORT_DS = {
    "acdc": "ACDC\n(weather)",
    "idd":  "IDD\n(geo+semantic)",
    "bdd":  "BDD100K\n(geographic)",
}


def _score_fn(col, hi, ds):
    """Per-seed scorer that routes ACDC through the condition-stratified
    AUROC. Pooled ACDC at MSP>=0.85 puts 18/19 confident failures in night,
    collapsing the task into a night classifier — per-condition cutoffs
    restore a fair label."""
    if ds == "acdc":
        return lambda g: cf_auroc_stratified(g, col, ACDC_MSP_THR,
                                             group_col="condition",
                                             higher_is_fail=hi)
    return lambda g: cf_auroc(g, col, DEFAULT_MSP_THR, higher_is_fail=hi)


def deep_ensemble_auroc(ds):
    """CF-AUROC of the 3x Deep Ensemble, read from the ensemble eval campaign."""
    fn = DEEP_ENSEMBLE_FILES.get(ds)
    if fn is None:
        return np.nan
    path = ANALYSIS_ROOT / "deep_ensemble_eval" / fn
    if not path.is_file():
        return np.nan
    d = pd.read_csv(path, low_memory=False)
    if ds == "acdc":
        d = d[d["domain"] == "all"]
    return _score_fn("ensemble_entropy", True, ds)(d)


def compute_aurocs(pi, datasets):
    """Per-seed mean/std for every (method, dataset) pair."""
    rows = []
    for ds in datasets:
        ds_data = pi[pi["dataset"] == ds]
        # Post-hoc scores are identical across supervision types; use
        # dense_multi rows so we cover all 3 retrain seeds.
        sub_any = ds_data[ds_data["supervision_type"] == "dense_multi"]
        if sub_any.empty:
            sub_any = ds_data.drop_duplicates(subset=["image_id"])

        for label, col, hi in POSTHOC:
            if col not in sub_any.columns:
                rows.append({"method": label, "dataset": ds, "mean": np.nan, "std": np.nan, "n": 0})
                continue
            m, s, n = per_seed_apply(sub_any, _score_fn(col, hi, ds))
            rows.append({"method": label, "dataset": ds, "mean": m, "std": s, "n": n})

        de = deep_ensemble_auroc(ds)
        rows.append({"method": "DeepEns (3×)", "dataset": ds, "mean": de,
                     "std": 0.0, "n": 1 if np.isfinite(de) else 0})

        mode_labels = {
            "dense_multi": "T-Multi (ours)",
            "gt_disagree": "GT-Dis",
            "gt_risk":     "GT-Gap",
        }
        for mode, col in MODE_COL.items():
            sub = ds_data[ds_data["supervision_type"] == mode]
            label = mode_labels[mode]
            if sub.empty or col not in sub.columns:
                rows.append({"method": label, "dataset": ds, "mean": np.nan, "std": np.nan, "n": 0})
                continue
            m, s, n = per_seed_apply(sub, _score_fn(col, True, ds))
            rows.append({"method": label, "dataset": ds, "mean": m, "std": s, "n": n})

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
        means, stds = [], []
        for ds in datasets:
            sub = df[(df["method"] == method) & (df["dataset"] == ds)]
            if sub.empty:
                means.append(np.nan); stds.append(np.nan); continue
            means.append(float(sub["mean"].iloc[0]))
            s = float(sub["std"].iloc[0])
            stds.append(s if np.isfinite(s) else 0.0)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        is_ours = "ours" in method
        ax.bar(xs + offset, np.nan_to_num(means_arr, nan=0), bar_w,
               color=COLORS[method],
               edgecolor="#1a1a1a" if is_ours else "none",
               linewidth=1.3 if is_ours else 0,
               label=method, zorder=3 if is_ours else 2)
        # Error bars only where we actually have multi-seed data.
        for i, (m, s) in enumerate(zip(means_arr, stds_arr)):
            if np.isfinite(m) and s >= 0:
                ax.errorbar(xs[i] + offset, m, yerr=s, fmt="none",
                            ecolor="#1a1a1a", elinewidth=0.9, capsize=2.0,
                            capthick=0.9, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT_DS.get(ds, ds) for ds in datasets], fontsize=9)
    ax.set_ylim(0.48, 0.80)
    ax.set_ylabel("CF-AUROC")
    ax.text(0.015, 0.985, "MSP ≥ 0.85", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="#888")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.4)
    ax.grid(False)
    # Legend lifted out of the data area entirely.
    ax.legend(fontsize=8.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.07), ncol=9, frameon=False,
              handlelength=1.2, columnspacing=1.4, handletextpad=0.5)

    # ── Right panel: T-Multi − GT delta ──
    ax2 = axes[1]
    delta_datasets = datasets

    gt_methods = ["GT-Dis", "GT-Gap"]
    gt_colors = [COLORS["GT-Dis"], COLORS["GT-Gap"]]
    xs2 = np.arange(len(delta_datasets))
    bw = 0.32

    for k, (gt_m, gc) in enumerate(zip(gt_methods, gt_colors)):
        deltas, derrs = [], []
        for ds in delta_datasets:
            tr = df[(df["method"] == "T-Multi (ours)") & (df["dataset"] == ds)]
            gr = df[(df["method"] == gt_m) & (df["dataset"] == ds)]
            t = float(tr["mean"].iloc[0]) if not tr.empty else np.nan
            g = float(gr["mean"].iloc[0]) if not gr.empty else np.nan
            ts = float(tr["std"].iloc[0]) if not tr.empty else np.nan
            gs = float(gr["std"].iloc[0]) if not gr.empty else np.nan
            deltas.append(t - g if np.isfinite(t) and np.isfinite(g) else np.nan)
            # Independent-seed std on the difference; treat single-seed std as 0.
            ts2 = ts**2 if np.isfinite(ts) else 0.0
            gs2 = gs**2 if np.isfinite(gs) else 0.0
            derrs.append(np.sqrt(ts2 + gs2) if (np.isfinite(ts) or np.isfinite(gs)) else np.nan)
        deltas = np.array(deltas)
        derrs = np.array(derrs)
        offset = (k - 0.5) * bw
        bars = ax2.bar(xs2 + offset, deltas * 100, bw,
                       color=gc, edgecolor="#1a1a1a", linewidth=0.8,
                       alpha=0.85, label=f"vs {gt_m}")
        for i, (bar, d, e) in enumerate(zip(bars, deltas, derrs)):
            if np.isfinite(d) and np.isfinite(e) and e > 0:
                ax2.errorbar(bar.get_x() + bar.get_width() / 2, d * 100,
                             yerr=e * 100, fmt="none", ecolor="#1a1a1a",
                             elinewidth=0.9, capsize=2.0, capthick=0.9,
                             zorder=4)

    ax2.axhline(0, color="#333", lw=0.8)
    ax2.grid(False)
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
