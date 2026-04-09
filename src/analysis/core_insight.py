"""
Figure: Student confidence is decorrelated from teacher benefit.

Left panel:  MSP vs teacher benefit (near-zero correlation)
Right panel: Guardrail++ vs teacher benefit (learned signal)

Usage:
    python src/analysis/core_insight_figure.py paper_eval_v3/csv
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = sys.argv[1] if len(sys.argv) > 1 else "/Users/shay/PycharmProjects/Guardrail-Distillation/src/analysis/cs_b0_b2_eval/csv"

pi = pd.read_csv(os.path.join(BASE, "per_image.csv"))

r_msp, _ = spearmanr(pi["student_msp"], pi["teacher_benefit"])
r_guard, _ = spearmanr(pi["guardrail_risk"], pi["teacher_benefit"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=200)

# ── Panel A: MSP vs teacher benefit ──
sc1 = ax1.scatter(
    pi["student_msp"], pi["teacher_benefit"],
    c=pi["student_risk"], cmap="RdYlGn_r", s=14, alpha=0.55, edgecolors="none",
)
ax1.set_xlabel("Student MSP (confidence)")
ax1.set_ylabel("Teacher benefit")
ax1.set_title("A.  Confidence (MSP)")
ax1.text(
    0.03, 0.97, f"Spearman ρ = {r_msp:+.3f}\nn = {len(pi)}",
    transform=ax1.transAxes, va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
)
plt.colorbar(sc1, ax=ax1, label="Student risk", shrink=0.82, pad=0.02)

# ── Panel B: Guardrail vs teacher benefit ──
sc2 = ax2.scatter(
    pi["guardrail_risk"], pi["teacher_benefit"],
    c=pi["student_risk"], cmap="RdYlGn_r", s=14, alpha=0.55, edgecolors="none",
)
ax2.set_xlabel("Guardrail++ predicted utility")
ax2.set_ylabel("Teacher benefit")
ax2.set_title("B.  Guardrail++")
ax2.text(
    0.03, 0.97, f"Spearman ρ = {r_guard:+.3f}",
    transform=ax2.transAxes, va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
)
plt.colorbar(sc2, ax=ax2, label="Student risk", shrink=0.82, pad=0.02)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Student confidence is decorrelated from teacher benefit",
    fontsize=14, y=1.01,
)
fig.tight_layout()

out = os.path.join(BASE, "figure_core_insight.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")