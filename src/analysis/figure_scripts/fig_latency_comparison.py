"""Inference latency comparison from measured data.

Horizontal bar chart showing real latency measurements from per_image.csv.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _lib import apply_style, load_table, savefig, pool_acdc_domain


def main():
    apply_style()

    pi = load_table("per_image")
    pi = pi[(pi["backbone"] == "b1") & (pi["supervision_type"] == "dense_multi")].copy()
    pi = pool_acdc_domain(pi)

    s_med = pd.to_numeric(pi["student_latency_ms"], errors="coerce").dropna().median()
    g_med = pd.to_numeric(pi["guardrail_latency_ms"], errors="coerce").dropna().median()
    mc_med = pd.to_numeric(pi["mc_dropout_latency_ms"], errors="coerce").dropna()
    mc_med = mc_med[mc_med > 0].median()
    t_med = pd.to_numeric(pi["teacher_latency_ms"], errors="coerce").dropna().median()

    methods = [
        "Student",
        "Student +\nGuardrail (ours)",
        "MC-Dropout\n(4 passes)",
        "Teacher\n(deferral)",
    ]
    latencies = [s_med, s_med + g_med, mc_med, t_med]
    colors = ["#888888", "#8172B3", "#4C72B0", "#937860"]

    fig, ax = plt.subplots(figsize=(4.5, 2.0))

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, latencies, height=0.55, color=colors,
                   edgecolor="white", linewidth=0.5)

    for bar, lat in zip(bars, latencies):
        w = bar.get_width()
        label = f"{lat:.1f} ms"
        if w > max(latencies) * 0.4:
            ax.text(w - 0.3, bar.get_y() + bar.get_height() / 2,
                    label, ha="right", va="center", fontsize=8,
                    fontweight="bold", color="white")
        else:
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    label, ha="left", va="center", fontsize=8,
                    fontweight="bold", color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_xlabel("Inference latency (ms, mit-b1, median)", fontsize=9)
    ax.set_xlim(0, max(latencies) * 1.25)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.grid(axis="y", visible=False)

    plt.tight_layout()
    savefig(fig, "fig_latency_comparison")


if __name__ == "__main__":
    main()
