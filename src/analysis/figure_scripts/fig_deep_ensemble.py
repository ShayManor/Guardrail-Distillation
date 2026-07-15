"""Risk-coverage figure: single-pass guardrail vs 3x Deep Ensemble (ACDC, mit-b1).

Self-contained (does not use combined_all, which lacks ensemble_entropy).
Data: ../deep_ensemble_eval/b1_per_image.csv (all methods scored on identical images).
Output: ../figures/fig_deep_ensemble.png (used as Fig. deepens-rc in the paper).
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "deep_ensemble_eval", "b1_per_image.csv")
OUT = os.path.join(ROOT, "figures", "fig_deep_ensemble.png")


def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


rows = [r for r in csv.DictReader(open(DATA)) if r.get("domain") == "all"]


def rc_curve(col, hi, npts=100):
    s = [r for r in rows
         if not math.isnan(f(r.get(col, "nan"))) and not math.isnan(f(r["student_risk"]))]
    sc = np.array([f(r[col]) for r in s])
    rk = np.array([f(r["student_risk"]) for r in s])
    order = np.argsort(sc) if hi else np.argsort(-sc)   # most-confident first
    sr = rk[order]
    cum = np.cumsum(sr) / (np.arange(len(sr)) + 1)
    n = len(sr)
    covs = np.linspace(1.0 / n, 1.0, npts)
    idx = np.clip((covs * n).astype(int) - 1, 0, n - 1)
    return covs, cum[idx]


def aurc(col, hi):
    c, r = rc_curve(col, hi, 200)
    return float(np.trapezoid(r, c))


plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.18, "grid.linestyle": "-", "figure.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight"})

# method: (col, higher_is_fail, label, color, linestyle)
METHODS = [
    ("student_msp", False, "MSP", "#4C72B0", "-"),
    ("energy_score", True, "Energy", "#CCB974", "-"),
    ("ensemble_entropy", True, "Deep Ensemble (3×)", "#C44E52", "--"),
    ("guardrailpp_utility_dense_gap", True, "T-Multi (ours, 1×)", "#8172B3", "-"),
]

fig, ax = plt.subplots(figsize=(5.2, 3.9))
for col, hi, lab, color, ls in METHODS:
    c, r = rc_curve(col, hi)
    ax.plot(c, r, label=f"{lab}  (AURC {aurc(col, hi):.3f})",
            color=color, ls=ls, lw=2.0)
ax.set_xlabel("Coverage")
ax.set_ylabel("Selective risk")
ax.set_xlim(0, 1)
# legend below the axes (2 columns) so it never overlaps the curves
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
          frameon=False, columnspacing=1.6, handlelength=1.9, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
for col, hi, lab, _, _ in METHODS:
    print(f"  {lab:22s} AURC={aurc(col, hi):.4f}")
