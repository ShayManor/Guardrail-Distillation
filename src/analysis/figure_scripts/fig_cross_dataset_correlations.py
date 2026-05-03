"""F17 — Cross-dataset correlation table: the negative result replicates.

Horizontal grouped bar chart showing key correlations across all 3 datasets:
  - rho(SR, TR): student-teacher risk coupling (should be high everywhere)
  - rho(dense-gap, benefit): does the head predict benefit? (should be near 0)
  - rho(dense-gap, student_risk): is it a risk proxy? (should be moderate-high)

Plus a panel showing surviving variance fraction: Var(benefit) / max(Var(SR), Var(TR)).

This proves the negative result is structural (not ACDC-specific) and the
mechanism (shift-robust risk proxy) generalizes across shift types.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, available_datasets, available_backbones,
    pool_acdc_domain, BACKBONE_COLORS, BACKBONE_LABELS, DATASET_LABELS,
)


def compute_correlations(sub):
    """Compute key correlations for a (dataset, backbone) subset."""
    sr = pd.to_numeric(sub["student_risk"], errors="coerce")
    tr = pd.to_numeric(sub["teacher_risk"], errors="coerce")
    tb = pd.to_numeric(sub["teacher_benefit"], errors="coerce")
    dg = pd.to_numeric(sub["guardrailpp_utility_dense_gap"], errors="coerce")
    msp = pd.to_numeric(sub["student_msp"], errors="coerce")

    mask = sr.notna() & tr.notna() & tb.notna() & dg.notna() & msp.notna()
    sr, tr, tb, dg, msp = sr[mask], tr[mask], tb[mask], dg[mask], msp[mask]

    if len(sr) < 10:
        return None

    var_sr = float(sr.var(ddof=1))
    var_tr = float(tr.var(ddof=1))
    var_tb = float(tb.var(ddof=1))
    surv = var_tb / max(var_sr, var_tr) if max(var_sr, var_tr) > 0 else np.nan

    return {
        "rho_sr_tr": float(sr.corr(tr)),
        "rho_neg_msp_benefit": float((-msp).corr(tb)),
        "rho_dg_benefit": float(dg.corr(tb)),
        "rho_dg_sr": float(dg.corr(sr)),
        "surviving_frac": surv,
        "n": len(sr),
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
            # Take first seed to avoid duplicate images
            seed0 = sub["seed"].min()
            sub = sub[sub["seed"] == seed0]
            corrs = compute_correlations(sub)
            if corrs is None:
                continue
            rows.append({"dataset": ds, "backbone": bb, **corrs})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # -- Left: correlation bars --
    ax = axes[0]
    metrics = [
        ("rho_sr_tr",           "rho(SR, TR)",        "#C44E52"),
        ("rho_dg_benefit",      "rho(dense-gap, benefit)", "#DD8452"),
        ("rho_neg_msp_benefit", "rho(-MSP, benefit)", "#999999"),
        ("rho_dg_sr",           "rho(dense-gap, SR)", "#4C72B0"),
    ]

    group_labels = []
    for ds in datasets:
        for bb in available_backbones(df[df["dataset"] == ds]):
            group_labels.append(f"{ds.upper()}\n{BACKBONE_LABELS[bb]}")

    xs = np.arange(len(group_labels))
    n_metrics = len(metrics)
    w = 0.75 / n_metrics

    for mi, (col, label, color) in enumerate(metrics):
        offset = (mi - (n_metrics - 1) / 2) * w
        vals = []
        for ds in datasets:
            for bb in available_backbones(df[df["dataset"] == ds]):
                row = df[(df["dataset"] == ds) & (df["backbone"] == bb)]
                vals.append(float(row[col].iloc[0]) if not row.empty else np.nan)
        ax.bar(xs + offset, vals, w, color=color, label=label, edgecolor="none")

    ax.axhline(0, color="#333", lw=0.8)
    ax.axhspan(-0.15, 0.15, color="red", alpha=0.04)
    ax.set_xticks(xs, group_labels, fontsize=7.5)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-0.3, 1.0)
    ax.set_title("Key correlations: negative result replicates across benchmarks")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    # Vertical separators between datasets
    idx = 0
    prev_ds = None
    for ds in datasets:
        n_bb = len(available_backbones(df[df["dataset"] == ds]))
        if prev_ds is not None:
            ax.axvline(idx - 0.5, color="#ccc", lw=0.8)
        prev_ds = ds
        idx += n_bb

    # -- Right: surviving variance fraction --
    ax = axes[1]
    group_labels2 = []
    surv_vals = []
    bar_colors = []
    for ds in datasets:
        for bb in available_backbones(df[df["dataset"] == ds]):
            group_labels2.append(f"{ds.upper()}\n{BACKBONE_LABELS[bb]}")
            row = df[(df["dataset"] == ds) & (df["backbone"] == bb)]
            surv_vals.append(float(row["surviving_frac"].iloc[0]) * 100 if not row.empty else np.nan)
            bar_colors.append(BACKBONE_COLORS.get(bb, "#999"))

    xs2 = np.arange(len(group_labels2))
    bars = ax.bar(xs2, surv_vals, 0.55, color=bar_colors, edgecolor="#1a1a1a", linewidth=0.8)

    for bar, v in zip(bars, surv_vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Vertical separators
    idx = 0
    prev_ds = None
    for ds in datasets:
        n_bb = len(available_backbones(df[df["dataset"] == ds]))
        if prev_ds is not None:
            ax.axvline(idx - 0.5, color="#ccc", lw=0.8)
        prev_ds = ds
        idx += n_bb

    ax.set_xticks(xs2, group_labels2, fontsize=7.5)
    ax.set_ylabel("Var(benefit) / max(Var(SR), Var(TR))  (%)")
    ax.set_title("Surviving variance: benefit signal is structurally weak")
    ax.set_ylim(0, 55)

    fig.suptitle("The negative result holds across all four benchmarks:\n"
                 "teacher_benefit is structurally unpredictable from student features",
                 fontsize=11, y=1.02)
    plt.subplots_adjust(wspace=0.24, top=0.84, bottom=0.14)
    savefig(fig, "fig_cross_dataset_correlations")


if __name__ == "__main__":
    main()
