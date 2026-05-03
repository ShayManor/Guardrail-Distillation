"""F8 — Latency / compute cost.

Left panel: teacher / student latency ratio per backbone per dataset (all
datasets with data). Right panel: absolute latency breakdown for b1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, available_datasets, available_backbones,
    pool_acdc_domain, BACKBONE_LABELS, BACKBONE_COLORS, DATASET_LABELS,
)

DATASET_BAR_COLORS = {"city": "#aaa", "acdc": "#4C72B0", "idd": "#55A868", "bdd": "#DD8452"}


def mean_ms(pi, bb, ds, col):
    sub = pi[(pi["backbone"] == bb) & (pi["dataset"] == ds)]
    v = pd.to_numeric(sub[col], errors="coerce")
    v = v[v.notna() & (v > 0)]
    return float(v.mean()) if len(v) else np.nan


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["supervision_type"] == "dense_multi"].copy()
    pi = pool_acdc_domain(pi)

    datasets = available_datasets(pi)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # -- Left: teacher / student ratio --
    ax = axes[0]
    bbs = ("b0", "b1", "b2")
    xs = np.arange(len(bbs))
    n_ds = len(datasets)
    w = 0.8 / max(n_ds, 1)

    for di, ds in enumerate(datasets):
        ratios = []
        for bb in bbs:
            s = mean_ms(pi, bb, ds, "student_latency_ms")
            t = mean_ms(pi, bb, ds, "teacher_latency_ms")
            ratios.append(t / s if np.isfinite(s) and np.isfinite(t) and s > 0 else np.nan)
        offset = (di - (n_ds - 1) / 2) * w
        color = DATASET_BAR_COLORS.get(ds, "#999")
        bars = ax.bar(xs + offset, ratios, width=w,
                      color=color, alpha=0.85,
                      edgecolor="#1a1a1a", linewidth=0.8,
                      label=DATASET_LABELS.get(ds, ds))
        for xi, (bar, r) in enumerate(zip(bars, ratios)):
            if not np.isfinite(r):
                continue
            ax.text(bar.get_x() + bar.get_width()/2, r + 0.04,
                    f"{r:.1f}x", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xs, [BACKBONE_LABELS[b] for b in bbs])
    ax.set_ylabel("Teacher / student latency ratio")
    ax.axhline(1.0, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.set_title("How expensive is deferring to the teacher?")
    ax.set_ylim(0, 6.5)
    ax.legend(title="dataset", fontsize=7.5, loc="upper right", frameon=False)

    # -- Right: guardrail vs student (b1 measured, pick dataset with most data) --
    ax = axes[1]
    # Prefer ACDC for consistency, fall back to first available
    ref_ds = "acdc" if "acdc" in datasets else datasets[0]
    s_ms = mean_ms(pi, "b1", ref_ds, "student_latency_ms")
    g_ms = mean_ms(pi, "b1", ref_ds, "guardrail_latency_ms")
    t_ms = mean_ms(pi, "b1", ref_ds, "teacher_latency_ms")

    vals = [s_ms, g_ms, t_ms]
    labels_x = ["student\nforward", "guardrail\nhead", "teacher\nforward"]
    colors_x = ["#4C72B0", "#8172B3", "#937860"]

    bars = ax.bar(labels_x, vals, color=colors_x, edgecolor="#1a1a1a", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                    f"{v:.2f} ms", ha="center", va="bottom", fontsize=9)

    if np.isfinite(s_ms) and np.isfinite(g_ms):
        ratio = g_ms / s_ms
        ax.text(0.5, 0.9,
                f"guardrail overhead ~ {ratio*100:.0f}% of student\nteacher ~ {t_ms/s_ms:.1f}x student",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                          edgecolor="#aaa", linewidth=0.6))

    ax.set_ylabel(f"Latency (ms, mit-b1, {ref_ds.upper()})")
    ax.set_title("Guardrail head adds one student-forward")
    ax.set_ylim(0, max(vals) * 1.35 if any(np.isfinite(v) for v in vals) else 1)

    plt.subplots_adjust(wspace=0.28, top=0.88, bottom=0.14)
    savefig(fig, "fig_latency")


if __name__ == "__main__":
    main()
