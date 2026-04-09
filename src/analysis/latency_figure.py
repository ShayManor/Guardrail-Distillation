"""
Figure: Cost-quality Pareto under teacher compute budget.

Left panel:  Effective system mIoU vs teacher budget %
Right panel: Fraction of total teacher benefit recovered vs budget %

Usage:
    python src/analysis/teacher_budget_figure.py paper_eval_v3/csv
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else "/Users/shay/PycharmProjects/Guardrail-Distillation/src/analysis/cs_b0_b2_eval/csv"

tb = pd.read_csv(os.path.join(BASE, "teacher_budget.csv"))

METHODS = {
    "guardrail":           ("Guardrail++",  "#8172B3", "-",  2.4),
    "msp":                 ("MSP",          "#4C72B0", "-",  1.4),
    "entropy":             ("Entropy",      "#DD8452", "-",  1.4),
    "mc_dropout":          ("MC-Dropout",   "#C44E52", "-",  1.4),
    "oracle":              ("Oracle",       "#937860", "--", 1.4),
    "random":              ("Random",       "#CCCCCC", "-.", 1.2),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)

for method, (label, color, ls, lw) in METHODS.items():
    sub = tb[tb["method"] == method].sort_values("teacher_budget")
    if sub.empty:
        continue
    ax1.plot(
        sub["teacher_budget"] * 100, sub["effective_miou"], ls,
        color=color, label=label, linewidth=lw, marker=".", markersize=3,
    )
    ax2.plot(
        sub["teacher_budget"] * 100, sub["benefit_recovered_frac"] * 100, ls,
        color=color, label=label, linewidth=lw, marker=".", markersize=3,
    )

ax1.set_xlabel("Teacher budget (% frames deferred)")
ax1.set_ylabel("Effective system mIoU")
ax1.set_title("A.  Effective quality under compute budget")
ax1.legend(fontsize=8, loc="lower right", frameon=True, fancybox=True, framealpha=0.9)

ax2.set_xlabel("Teacher budget (% frames deferred)")
ax2.set_ylabel("Teacher benefit recovered (%)")
ax2.set_title("B.  Budget efficiency")
ax2.legend(fontsize=8, loc="lower right", frameon=True, fancybox=True, framealpha=0.9)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", alpha=0.2)

fig.suptitle(
    "Cost-Quality Pareto",
    fontsize=13, y=1.01,
)
fig.tight_layout()

out = os.path.join(BASE, "figure_teacher_budget.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")