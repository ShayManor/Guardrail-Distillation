"""Confident failure detection: AUROC across MSP thresholds. Input: CSV dir (first arg)."""
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else "/Users/shay/PycharmProjects/Guardrail-Distillation/src/analysis/cs_b0_b2_eval/csv"
cf = pd.read_csv(os.path.join(BASE, "confident_failures.csv"))

COLORS = {"msp": "#4C72B0", "entropy": "#DD8452", "mc_dropout": "#C44E52",
          "guardrail": "#8172B3", "oracle": "#937860"}

thresholds = cf["msp_threshold"].values
detectors = {
    "MSP":         ("msp_auroc",        COLORS["msp"],        "s", 1.6, "-"),
    "Entropy":     ("entropy_auroc",    COLORS["entropy"],    "s", 1.6, "-"),
    "MC Dropout":  ("mc_dropout_auroc", COLORS["mc_dropout"], "s", 1.6, "-"),
    "Guardrail++": ("guardrail_auroc",  COLORS["guardrail"],  "o", 2.5, "-"),
    "Oracle":      ("oracle_auroc",     COLORS["oracle"],     "D", 1.4, "--"),
}

fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)

for label, (col, color, marker, lw, ls) in detectors.items():
    vals = cf[col].values
    ax.plot(thresholds, vals, ls, color=color, label=label,
            linewidth=lw, marker=marker, markersize=6 if label == "Guardrail++" else 5)

ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
ax.set_xlabel("MSP confidence threshold")
ax.set_ylabel("AUROC for detecting failures")
ax.set_title("Guardrail++ advantage widens at high confidence",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(0.45, 0.75)

# Annotate headline result at MSP >= 0.97
g97 = float(cf[cf["msp_threshold"] == 0.97]["guardrail_auroc"].iloc[0])
m97 = float(cf[cf["msp_threshold"] == 0.97]["msp_auroc"].iloc[0])
delta = g97 - m97
ax.annotate(f"Δ = {delta:+.4f}",
            xy=(0.97, g97), xytext=(0.925, 0.69),
            fontsize=9, fontweight="bold", color=COLORS["guardrail"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["guardrail"]))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.2)
fig.tight_layout()

out = os.path.join(BASE, "figure_confident_failures.png")
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)