"""
ACDC Domain Shift Figures for PI Presentation
==============================================
Strengthens Contribution 1 (teacher gap ≠ student uncertainty under shift)
and motivates Contributions 2-3.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os, sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "/Users/shay/PycharmProjects/Guardrail-Distillation/src/analysis/acdc_b0_b2_eval/csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/Users/shay/PycharmProjects/Guardrail-Distillation/src/analysis/acdc_b0_b2_eval/csv"
os.makedirs(OUT, exist_ok=True)

pi = pd.read_csv(os.path.join(BASE, "per_image.csv"))
runs = pd.read_csv(os.path.join(BASE, "runs.csv"))
cf = pd.read_csv(os.path.join(BASE, "confident_failures.csv"))
rc = pd.read_csv(os.path.join(BASE, "risk_coverage.csv"))
tb = pd.read_csv(os.path.join(BASE, "teacher_budget.csv"))

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "figure.facecolor": "white",
})

COND_COLORS = {"fog": "#8DA0CB", "night": "#1B1B2F", "rain": "#66C2A5", "snow": "#FC8D62"}
COND_ORDER = ["fog", "rain", "snow", "night"]
COND_LABELS = {"fog": "Fog", "rain": "Rain", "snow": "Snow", "night": "Night"}

# Cityscapes reference values (from previous v3 eval)
CS_STUDENT_MIOU = 0.537
CS_TEACHER_MIOU = 0.630
CS_TEACHER_BENEFIT = 0.095
CS_MSP_RHO = -0.114

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Domain shift impact — mIoU collapses, teacher benefit rises
# Strengthens: Contribution 1 (gap ≠ uncertainty) + Contribution 3 (need for deferral)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

# Panel A: mIoU comparison
ax = axes[0]
conds = COND_ORDER
x = np.arange(len(conds) + 1)
# Calculate per-condition statistics from per_image data
cond_stats = pi.groupby("condition").agg({"student_miou": "mean", "teacher_miou": "mean"})
student_vals = [CS_STUDENT_MIOU] + [float(cond_stats.loc[c, "student_miou"]) for c in conds]
teacher_vals = [CS_TEACHER_MIOU] + [float(cond_stats.loc[c, "teacher_miou"]) for c in conds]
labels = ["Cityscapes\n(in-dist)"] + [COND_LABELS[c] for c in conds]

w = 0.35
b1 = ax.bar(x - w/2, student_vals, w, label="Student (MiT-B0)", color="#8172B3")
b2 = ax.bar(x + w/2, teacher_vals, w, label="Teacher (B5)", color="#55A868")
for bars in [b1, b2]:
    for bar, val in zip(bars, student_vals if bars is b1 else teacher_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("mIoU")
ax.set_title("A. Model quality degrades under domain shift", fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.75)
ax.axvline(0.5, color="gray", ls="--", alpha=0.3)

# Panel B: Teacher benefit comparison
ax = axes[1]
benefit_stats = pi.groupby("condition")["teacher_benefit"].mean()
benefits = [CS_TEACHER_BENEFIT] + [float(benefit_stats.loc[c]) for c in conds]
colors = ["#999999"] + [COND_COLORS[c] for c in conds]
bars = ax.bar(x, benefits, 0.55, color=colors)
for bar, val in zip(bars, benefits):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Mean teacher benefit (Δ risk)")
ax.set_title("B. Teacher benefit increases under shift", fontsize=12)
ax.axhline(CS_TEACHER_BENEFIT, color="gray", ls=":", alpha=0.5)
ax.text(len(conds) + 0.3, CS_TEACHER_BENEFIT + 0.002, "Cityscapes baseline", fontsize=8, color="gray")

fig.suptitle("Distribution shift amplifies the need for intelligent deferral",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_shift_impact.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"[1/4] {OUT}/fig1_shift_impact.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Confident failure explosion under shift
# Strengthens: Contribution 1 (MSP is blind to failures)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

fail_data = []
for cond in COND_ORDER:
    sub = pi[pi["condition"] == cond]
    conf = sub[sub["student_msp"] >= 0.85]
    # Use top-20% worst as failure (consistent with eval)
    cutoff = sub["student_risk"].quantile(0.80)
    fails = conf[conf["student_risk"] >= cutoff]
    # Also compute hard failures (mIoU < 0.3)
    hard_fails = conf[conf["student_miou"] <= 0.30]
    fail_data.append({
        "condition": COND_LABELS[cond],
        "n_confident": len(conf),
        "n_hard_fail": len(hard_fails),
        "hard_fail_pct": 100 * len(hard_fails) / max(len(conf), 1),
        "color": COND_COLORS[cond],
    })

fd = pd.DataFrame(fail_data)
x = np.arange(len(fd))
bars = ax.bar(x, fd["hard_fail_pct"], 0.55, color=fd["color"].tolist())
for i, (bar, row) in enumerate(zip(bars, fd.itertuples())):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{row.n_hard_fail}/{row.n_confident}\n({row.hard_fail_pct:.0f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(fd["condition"], fontsize=11)
ax.set_ylabel("Confident frames with mIoU ≤ 0.30 (%)", fontsize=11)
ax.set_title("Confident Failures (MSP ≥ 0.85)",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 115)
ax.axhline(0, color="gray", lw=0.5)

fig.tight_layout()
fig.savefig(f"{OUT}/fig2_confident_failures.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"[2/4] {OUT}/fig2_confident_failures.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: MSP ρ stays near zero across ALL conditions
# Strengthens: Contribution 1 — the decorrelation is universal
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

rho_data = []
for cond in COND_ORDER:
    sub = pi[pi["condition"] == cond]
    r_msp, _ = spearmanr(sub["student_msp"], sub["teacher_benefit"])
    r_mc, _ = spearmanr(sub["mc_entropy"], sub["teacher_benefit"])
    r_ent, _ = spearmanr(sub["student_entropy"], sub["teacher_benefit"])
    rho_data.append({"condition": COND_LABELS[cond], "MSP": -r_msp, "Entropy": r_ent,
                     "MC Dropout": r_mc, "color": COND_COLORS[cond]})

# Add Cityscapes
rho_data.insert(0, {"condition": "Cityscapes\n(in-dist)", "MSP": -CS_MSP_RHO,
                     "Entropy": 0.108, "MC Dropout": 0.115, "color": "#999999"})

rd = pd.DataFrame(rho_data)
x = np.arange(len(rd))
w = 0.25
b1 = ax.bar(x - w, rd["MSP"], w, label="MSP", color="#4C72B0")
b2 = ax.bar(x, rd["Entropy"], w, label="Entropy", color="#DD8452")
b3 = ax.bar(x + w, rd["MC Dropout"], w, label="MC Dropout", color="#C44E52")

for bars in [b1, b2, b3]:
    for bar in bars:
        val = bar.get_height()
        y = val + 0.01 if val >= 0 else val - 0.03
        ax.text(bar.get_x() + bar.get_width()/2, y, f"{val:.2f}",
                ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5)

ax.axhline(0, color="black", lw=0.8)
ax.axhspan(-0.15, 0.15, color="red", alpha=0.06)
ax.text(0.03, 0.03, "ρ ≈ 0", fontsize=9, color="red", alpha=0.7, ha="left", va="bottom",
        transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.5))

ax.set_xticks(x)
ax.set_xticklabels(rd["condition"], fontsize=10)
ax.set_ylabel("Spearman ρ with teacher benefit", fontsize=11)
ax.set_title("No uncertainty method correlates with teacher benefit",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(-0.35, 0.35)

fig.tight_layout()
fig.savefig(f"{OUT}/fig3_rho_by_condition.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"[3/4] {OUT}/fig3_rho_by_condition.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Per-condition scatter — MSP vs teacher benefit (4 panels)
# Strengthens: Contribution 1 — visual proof of decorrelation per condition
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)

for ax, cond in zip(axes.flat, COND_ORDER):
    sub = pi[pi["condition"] == cond]
    r, p = spearmanr(sub["student_msp"], sub["teacher_benefit"])
    sc = ax.scatter(sub["student_msp"], sub["teacher_benefit"],
                    c=sub["student_risk"], cmap="RdYlGn_r", s=20, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Student MSP (confidence)")
    ax.set_ylabel("Teacher benefit (Δ risk)")
    ax.set_title(f"{COND_LABELS[cond]}:  ρ = {r:.3f}   (n={len(sub)})", fontsize=11)
    # Highlight confident failures
    conf_fail = sub[(sub["student_msp"] >= 0.90) & (sub["student_miou"] <= 0.30)]
    if len(conf_fail) > 0:
        ax.scatter(conf_fail["student_msp"], conf_fail["teacher_benefit"],
                   facecolors="none", edgecolors="red", s=50, lw=1.5,
                   label=f"Confident failures ({len(conf_fail)})")
        ax.legend(fontsize=8)

fig.suptitle("MSP is decorrelated from teacher benefit across all adverse conditions",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_scatter_by_condition.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"[4/4] {OUT}/fig4_scatter_by_condition.png")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\nAll figures saved to {OUT}/")
