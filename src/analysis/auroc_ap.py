"""AURC comparison: Guardrail++ vs all baselines. Input: CSV dir (first arg)."""
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else "/Users/shay/PycharmProjects/Guardrail-Distillation/src/analysis/cs_b0_b2_eval/csv"
rc = pd.read_csv(os.path.join(BASE, "risk_coverage.csv"))

COLORS = {"msp": "#4C72B0", "entropy": "#DD8452", "temp_msp": "#55A868",
          "mc_dropout": "#C44E52", "guardrail": "#8172B3", "oracle": "#937860",
          "teacher_oracle": "#DA8BC3"}
LABELS = {"oracle": "Oracle", "guardrail": "Guardrail++", "temp_msp": "Temp-MSP",
          "msp": "MSP", "neg_entropy": "Entropy", "mc_dropout": "MC Dropout (8×)",
          "teacher_oracle": "Teacher Oracle"}

def get_color(m):
    for k, c in COLORS.items():
        if k in m:
            return c
    return "#333333"

aurc = rc.groupby("method")["aurc"].first().reset_index()
show = ["oracle", "guardrail", "temp_msp", "msp", "neg_entropy", "mc_dropout", "teacher_oracle"]
aurc = aurc[aurc["method"].isin(show)].copy()
aurc["label"] = aurc["method"].map(LABELS)
aurc = aurc.sort_values("aurc")

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
bars = ax.barh(range(len(aurc)), aurc["aurc"].values, height=0.58,
               color=[get_color(m) for m in aurc["method"].values])

for i, (_, row) in enumerate(aurc.iterrows()):
    ax.text(row["aurc"] + 0.0008, i, f"{row['aurc']:.4f}", va="center", fontsize=10)

ax.set_yticks(range(len(aurc)))
ax.set_yticklabels(aurc["label"].values, fontsize=10)
ax.set_xlabel("AURC")
ax.set_title("AURC Comparison",
             fontsize=12, fontweight="bold")

lo = aurc["aurc"].min() - 0.005
hi = aurc["aurc"].max() + 0.008
ax.set_xlim(lo, hi)

msp_v = float(aurc[aurc["method"] == "msp"]["aurc"].iloc[0])
grd_v = float(aurc[aurc["method"] == "guardrail"]["aurc"].iloc[0])
ax.text(0.97, 0.04, f"Δ vs MSP: {grd_v - msp_v:.4f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8E0F0", ec="0.7"))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()

out = os.path.join(BASE, "figure_aurc_comparison.png")
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)