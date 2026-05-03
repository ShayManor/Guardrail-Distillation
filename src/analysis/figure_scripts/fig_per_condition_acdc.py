"""F7 — Per-condition ACDC: where does the dense head actually help?

Single panel. For each ACDC condition (fog/night/rain/snow) we compute
confident-failure AUROC for MSP and for the dense-gap head on every
backbone, then report the mean across backbones with a min/max range bar.
Averaging across backbones removes the b0/b1/b2 idiosyncrasies that made
the old 3-subplot view look noisy, and exposes the real story: night is
by far the hardest condition under shift (MSP near chance) and that's
exactly where the dense head lifts AUROC the most.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import apply_style, load_table, savefig, cf_auroc, METHOD_COLORS

CONDS = ("fog", "night", "rain", "snow")
BACKBONES = ("b0", "b1", "b2")


def per_cond_auroc(pi, bb, cond, score_col, higher=True, thr=0.85):
    sub = pi[(pi["backbone"] == bb) & (pi["dataset"] == "acdc") & (pi["domain"] == cond)]
    if sub.empty or score_col not in sub.columns:
        return float("nan")
    return cf_auroc(sub, score_col, thr, higher_is_fail=higher)


def agg(pi, cond, score_col, higher):
    vals = [per_cond_auroc(pi, bb, cond, score_col, higher=higher) for bb in BACKBONES]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["supervision_type"] == "dense_multi"].copy()
    if "domain" in pi.columns:
        pi = pi[pi["domain"].isin(CONDS)]

    msp_mean, msp_lo, msp_hi = [], [], []
    dg_mean, dg_lo, dg_hi = [], [], []
    for c in CONDS:
        m, lo, hi = agg(pi, c, "student_msp", higher=False)
        msp_mean.append(m); msp_lo.append(lo); msp_hi.append(hi)
        m, lo, hi = agg(pi, c, "guardrailpp_utility_dense_gap", higher=True)
        dg_mean.append(m); dg_lo.append(lo); dg_hi.append(hi)

    msp_mean = np.array(msp_mean); msp_lo = np.array(msp_lo); msp_hi = np.array(msp_hi)
    dg_mean = np.array(dg_mean); dg_lo = np.array(dg_lo); dg_hi = np.array(dg_hi)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    xs = np.arange(len(CONDS))
    w = 0.36

    msp_err = np.vstack([msp_mean - msp_lo, msp_hi - msp_mean])
    dg_err = np.vstack([dg_mean - dg_lo, dg_hi - dg_mean])

    ax.bar(xs - w/2, msp_mean, width=w, color=METHOD_COLORS["msp"],
           yerr=msp_err, capsize=3, ecolor="#555",
           label="MSP (baseline)", edgecolor="none")
    ax.bar(xs + w/2, dg_mean, width=w, color=METHOD_COLORS["dense_gap"],
           yerr=dg_err, capsize=3, ecolor="#333",
           label="dense-gap (ours)",
           edgecolor="#1a1a1a", linewidth=0.9)

    for i in range(len(CONDS)):
        if np.isfinite(msp_mean[i]) and np.isfinite(dg_mean[i]):
            delta = dg_mean[i] - msp_mean[i]
            color = "#1b7a28" if delta > 0 else "#a6201a"
            top = max(dg_hi[i], msp_hi[i])
            ax.text(xs[i], top + 0.035, f"Δ={delta:+.2f}",
                    ha="center", va="bottom", fontsize=9.5, color=color,
                    fontweight="bold")

    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.text(len(CONDS) - 0.55, 0.51, "chance", fontsize=8, color="gray")
    ax.set_xticks(xs, [c.capitalize() for c in CONDS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Confident-fail AUROC @ msp ≥ 0.85")
    ax.set_title("ACDC per-condition: dense-gap lifts the hardest shift (night)\n"
                 "mean over mit-b0/b1/b2, bars = min/max across backbones",
                 fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    plt.subplots_adjust(bottom=0.13, top=0.84, left=0.10, right=0.97)
    savefig(fig, "fig_per_condition_acdc")


if __name__ == "__main__":
    main()
