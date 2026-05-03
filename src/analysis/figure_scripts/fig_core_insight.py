"""F6 — "Why scalar teacher_benefit is unpredictable".

Three-panel insight:
  (A) student_risk vs teacher_risk scatter, r~0.8 regression line overlaid.
  (B) dense_gap predicted utility vs teacher_benefit scatter. Near-zero
      correlation shows the head does NOT recover benefit directly.
  (C) Same dense_gap utility vs student_risk scatter. Strong positive
      correlation shows the head ranks images by expected risk.

Default backbone = b1, default dataset = ACDC pooled-all. Override with
GD_BACKBONE / GD_DATASET env vars. Generates one figure per OOD dataset
if GD_DATASET is not set (ACDC + BDD side by side).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from _lib import (
    apply_style, load_table, savefig, pool_acdc_domain, available_datasets,
    DATASET_LABELS, BACKBONE_LABELS,
)


def _scatter_row(axes, sr, tr, tb, dg, bb, ds, n):
    """Draw the 3-panel scatter on a row of axes."""
    r_sr_tr, _ = pearsonr(sr, tr)
    r_dg_tb, _ = pearsonr(dg, tb)
    r_dg_sr, _ = pearsonr(dg, sr)

    # Panel A
    ax = axes[0]
    ax.scatter(sr, tr, c=sr, cmap="viridis", s=18, alpha=0.55, edgecolors="none")
    xs = np.linspace(sr.min(), sr.max(), 50)
    m, b = np.polyfit(sr, tr, 1)
    ax.plot(xs, m * xs + b, color="#c0392b", lw=1.6, linestyle="--")
    lim = max(sr.max(), tr.max()) * 1.02
    ax.plot([0, lim], [0, lim], color="#333", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("student_risk")
    ax.set_ylabel("teacher_risk")
    ax.set_title(f"Student vs teacher risk\nr = {r_sr_tr:+.2f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    # Panel B
    ax = axes[1]
    ax.scatter(dg, tb, c=sr, cmap="viridis", s=18, alpha=0.55, edgecolors="none")
    ax.axhline(0, color="#333", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("dense_gap utility (predicted)")
    ax.set_ylabel("teacher_benefit (actual)")
    ax.set_title(f"Does it predict benefit?\nr = {r_dg_tb:+.2f}  ->  NO")

    # Panel C
    ax = axes[2]
    ax.scatter(dg, sr, c=sr, cmap="viridis", s=18, alpha=0.55, edgecolors="none")
    m, b = np.polyfit(dg, sr, 1)
    xs = np.linspace(dg.min(), dg.max(), 50)
    ax.plot(xs, m * xs + b, color="#c0392b", lw=1.6, linestyle="--")
    ax.set_xlabel("dense_gap utility (predicted)")
    ax.set_ylabel("student_risk (actual)")
    ax.set_title(f"Does it rank risk?\nr = {r_dg_sr:+.2f}  ->  YES")


def main():
    apply_style()
    bb = os.environ.get("GD_BACKBONE", "b1")
    ds_override = os.environ.get("GD_DATASET", None)

    pi = load_table("per_image")
    pi = pi[(pi["supervision_type"] == "dense_multi") & (pi["backbone"] == bb)].copy()
    pi = pool_acdc_domain(pi)

    # Determine which datasets to show
    if ds_override:
        target_datasets = [ds_override]
    else:
        # Show all OOD datasets (skip cityscapes — it's boring for this plot)
        target_datasets = [ds for ds in available_datasets(pi) if ds != "city"]
        if not target_datasets:
            target_datasets = available_datasets(pi)[:1]

    n_rows = len(target_datasets)
    fig, all_axes = plt.subplots(n_rows, 3, figsize=(13.5, 4.6 * n_rows),
                                 squeeze=False)

    for row_i, ds in enumerate(target_datasets):
        sub = pi[pi["dataset"] == ds]
        if sub.empty:
            for ax in all_axes[row_i]:
                ax.set_visible(False)
            continue

        sr = pd.to_numeric(sub["student_risk"], errors="coerce")
        tr = pd.to_numeric(sub["teacher_risk"], errors="coerce")
        tb = pd.to_numeric(sub["teacher_benefit"], errors="coerce")
        dg = pd.to_numeric(sub["guardrailpp_utility_dense_gap"], errors="coerce")

        mask = sr.notna() & tr.notna() & tb.notna() & dg.notna()
        sr, tr, tb, dg = sr[mask], tr[mask], tb[mask], dg[mask]

        if len(sr) < 10:
            for ax in all_axes[row_i]:
                ax.set_visible(False)
            continue

        _scatter_row(all_axes[row_i], sr, tr, tb, dg, bb, ds, len(sr))

        # Row label
        ds_label = DATASET_LABELS.get(ds, ds)
        all_axes[row_i][0].annotate(
            f"{BACKBONE_LABELS[bb]}  /  {ds_label}  (n={len(sr)})",
            xy=(0.5, 1.18), xycoords="axes fraction",
            ha="center", fontsize=11, fontweight="bold")

    plt.subplots_adjust(wspace=0.32, hspace=0.45, top=0.88, bottom=0.10)
    savefig(fig, "fig_core_insight")


if __name__ == "__main__":
    main()
