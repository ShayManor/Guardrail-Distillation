"""F15 — Cross-dataset headline: dense-gap vs a post-hoc baseline.

Emits two figures:
  fig_cross_dataset_headline_msp     (MSP baseline)
  fig_cross_dataset_headline_energy  (Energy baseline)

Each figure has two panels:
  Left:  CF-AUROC @ msp>=0.85  (higher is better)
  Right: AURC delta in pp      (higher is better, ours - baseline negated)

Layout: x-axis grouped by dataset, inner ticks are just b0/b1/b2, with a
single dataset label under each group of three bars.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, compute_cf_auroc_table, compute_aurc_table,
    mean_std_across_seeds, available_datasets, available_backbones,
    BACKBONE_LABELS, DATASET_LABELS, METHOD_COLORS,
)

OURS = "dense_gap"
OURS_LABEL = "Dense-gap (ours)"


def _agg(df, ds, bb, method, col):
    sub = df[(df["dataset"] == ds) & (df["backbone"] == bb) & (df["method"] == method)]
    agg = mean_std_across_seeds(sub, ["backbone"], col)
    if agg.empty:
        return np.nan, 0.0
    mean = float(agg["mean"].iloc[0])
    std = agg["std"].iloc[0]
    return mean, float(std) if pd.notna(std) else 0.0


def _dataset_short(ds):
    return DATASET_LABELS.get(ds, ds).split("(")[0].strip()


def _group_positions(datasets, bbs_per_ds, group_pad=0.9, bar_pad=0.0):
    """Return (xs_per_cell, group_centers, ds_starts_ends)."""
    xs = []
    centers = []
    ranges = []
    cursor = 0.0
    for ds in datasets:
        n = len(bbs_per_ds[ds])
        start = cursor
        for _ in range(n):
            xs.append(cursor)
            cursor += 1.0 + bar_pad
        end = cursor - 1.0
        centers.append((start + end) / 2)
        ranges.append((start, end))
        cursor += group_pad
    return np.array(xs), centers, ranges


def _annotate_groups(ax, datasets, centers, ranges, y_frac=-0.16):
    """Put dataset labels below each group and draw soft underline."""
    trans = ax.get_xaxis_transform()
    for ds, c, (a, b) in zip(datasets, centers, ranges):
        ax.plot([a - 0.35, b + 0.35], [y_frac + 0.04, y_frac + 0.04],
                color="#bbb", lw=0.8, transform=trans, clip_on=False)
        ax.text(c, y_frac, _dataset_short(ds), ha="center", va="top",
                fontsize=10, fontweight="bold", transform=trans)


def _plot_one(baseline, baseline_label, suffix):
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["supervision_type"] == "dense_multi"].copy()

    cf = compute_cf_auroc_table(pi, msp_thresholds=(0.85,))
    cf = cf[cf["msp_threshold"] == 0.85]
    aurc = compute_aurc_table(pi)

    datasets = available_datasets(cf)
    bbs_per_ds = {ds: available_backbones(cf[cf["dataset"] == ds]) for ds in datasets}
    keys = [(ds, bb) for ds in datasets for bb in bbs_per_ds[ds]]

    xs, centers, ranges = _group_positions(datasets, bbs_per_ds)
    bb_labels = [bb.replace("mit-", "").upper() if bb.startswith("mit-") else bb[-2:].upper()
                 for _, bb in keys]

    base_a = np.array([_agg(cf, ds, bb, baseline, "auroc")[0] for ds, bb in keys])
    base_e = np.array([_agg(cf, ds, bb, baseline, "auroc")[1] for ds, bb in keys])
    ours_a = np.array([_agg(cf, ds, bb, OURS, "auroc")[0] for ds, bb in keys])
    ours_e = np.array([_agg(cf, ds, bb, OURS, "auroc")[1] for ds, bb in keys])

    base_aurc = np.array([_agg(aurc, ds, bb, baseline, "aurc")[0] for ds, bb in keys])
    ours_aurc = np.array([_agg(aurc, ds, bb, OURS, "aurc")[0] for ds, bb in keys])
    aurc_delta_pp = (base_aurc - ours_aurc) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    w = 0.38

    # ── Left: CF-AUROC grouped bars ──
    ax = axes[0]
    ax.bar(xs - w/2, base_a, w, yerr=base_e, capsize=2, ecolor="#666",
           color=METHOD_COLORS[baseline], label=baseline_label, edgecolor="none")
    ax.bar(xs + w/2, ours_a, w, yerr=ours_e, capsize=2, ecolor="#333",
           color=METHOD_COLORS[OURS], label=OURS_LABEL,
           edgecolor="#1a1a1a", linewidth=0.7)

    for i, x in enumerate(xs):
        if np.isfinite(base_a[i]) and np.isfinite(ours_a[i]):
            d = ours_a[i] - base_a[i]
            top = max(base_a[i] + base_e[i], ours_a[i] + ours_e[i])
            color = "#1b7a28" if d > 0.005 else "#a6201a" if d < -0.005 else "#666"
            ax.text(x, top + 0.018, f"{d:+.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold")

    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_xticks(xs, bb_labels, fontsize=8)
    ax.set_ylim(0.40, 1.02)
    ax.set_ylabel("CF-AUROC @ MSP >= 0.85")
    ax.set_title("Confident-failure detection", fontsize=11)
    handles, labels_ = ax.get_legend_handles_labels()
    _annotate_groups(ax, datasets, centers, ranges)

    # ── Right: AURC delta bars ──
    ax = axes[1]
    colors = ["#1b7a28" if d > 0.1 else "#a6201a" if d < -0.1 else "#999"
              for d in aurc_delta_pp]
    bars = ax.bar(xs, aurc_delta_pp, 0.62, color=colors,
                  edgecolor="#1a1a1a", linewidth=0.6)
    for bar, d in zip(bars, aurc_delta_pp):
        if not np.isfinite(d):
            continue
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        off = 0.08 if y >= 0 else -0.08
        ax.text(bar.get_x() + bar.get_width()/2, y + off,
                f"{d:+.1f}", ha="center", va=va, fontsize=7.5, fontweight="bold")

    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(xs, bb_labels, fontsize=8)
    ax.set_ylabel(f"AURC improvement over {baseline_label} (pp)")
    ax.set_title("Selective-prediction improvement", fontsize=11)
    _annotate_groups(ax, datasets, centers, ranges)

    fig.suptitle(f"Dense disagreement supervision vs {baseline_label}",
                 fontsize=12, x=0.02, y=0.97, ha="left")
    fig.legend(handles, labels_, fontsize=9, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), frameon=False,
               handlelength=1.6, ncol=1)
    plt.subplots_adjust(wspace=0.22, top=0.86, bottom=0.2)
    savefig(fig, f"fig_cross_dataset_headline_{suffix}")


def main():
    _plot_one("msp", "MSP", "msp")
    _plot_one("energy", "Energy", "energy")


if __name__ == "__main__":
    main()
