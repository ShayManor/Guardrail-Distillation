"""
Figure: Risk-coverage curves comparing all deferral methods.

Shows cumulative risk as a function of coverage. Lower is better.
Guardrail++ should dominate MSP/Entropy/MC-Dropout at every coverage level.

Usage:
    python src/analysis/risk_coverage_figure.py paper_eval_v3/csv
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else "/Users/shay/PycharmProjects/Guardrail-Distillation/src/analysis/cs_b0_b2_eval/csv"

rc = pd.read_csv(os.path.join(BASE, "risk_coverage.csv"))

METHODS = {
    "guardrail":       ("Guardrail++",     "#8172B3", "-",  2.4),
    "msp":             ("MSP",             "#4C72B0", "-",  1.4),
    "neg_entropy":     ("Entropy",         "#DD8452", "-",  1.4),
    "mc_dropout":      ("MC-Dropout",      "#C44E52", "-",  1.4),
    "teacher_oracle":  ("Teacher Oracle",  "#DA8BC3", "--", 1.4),
    "oracle":          ("Oracle",          "#937860", "--", 1.4),
}

fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=200)

for method, (label, color, ls, lw) in METHODS.items():
    sub = rc[rc["method"] == method].sort_values("coverage")
    if sub.empty:
        continue
    aurc = sub["aurc"].iloc[0]
    ax.plot(
        sub["coverage"], sub["risk"], ls,
        color=color, label=f"{label}  AURC {aurc:.4f}", linewidth=lw,
    )

ax.set_xlabel("Coverage (fraction of images retained)")
ax.set_ylabel("Cumulative risk (1 − mIoU)")
ax.set_title(
    "Risk-Coverage",
    fontsize=12,
)
ax.legend(fontsize=9, loc="lower right", frameon=True, fancybox=True, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="both", alpha=0.2)

fig.tight_layout()

out = os.path.join(BASE, "figure_risk_coverage.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")