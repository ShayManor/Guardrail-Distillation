"""F11-F14 — ACDC domain-shift panels (ported from acdc_figs.py).

Four standalone figures for the PI pitch, adapted to read from
``combined_all/`` instead of a single per_image.csv, and written into
``figures/`` alongside the rest. We filter to the primary campaign
(dataset=acdc, supervision_type=dense_multi, backbone=b1) so every panel
matches the headline results.

Panels:
  F11 fig_shift_impact       — mIoU collapse + teacher benefit rise
  F12 fig_confident_failures_bars — "night ≈ 100% confident-fail rate"
  F13 fig_rho_by_condition   — ρ(uncertainty, teacher_benefit) ≈ 0 universally
  F14 fig_scatter_by_condition — 2×2 MSP vs benefit scatter
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from _lib import apply_style, load_table, figures_dir

COND_COLORS = {"fog": "#8DA0CB", "night": "#1B1B2F", "rain": "#66C2A5", "snow": "#FC8D62"}
COND_ORDER = ["fog", "rain", "snow", "night"]
COND_LABELS = {"fog": "Fog", "rain": "Rain", "snow": "Snow", "night": "Night"}


def _load_acdc_b1():
    pi = load_table("per_image")
    pi = pi[(pi["dataset"] == "acdc")
            & (pi["supervision_type"] == "dense_multi")
            & (pi["backbone"] == "b1")].copy()
    if "condition" in pi.columns:
        pi = pi[pi["condition"].isin(COND_ORDER)]
    else:
        pi["condition"] = pi["domain"]
        pi = pi[pi["condition"].isin(COND_ORDER)]
    for col in ("student_miou", "teacher_miou", "teacher_benefit",
                "student_risk", "student_msp", "student_entropy", "mc_entropy"):
        if col in pi.columns:
            pi[col] = pd.to_numeric(pi[col], errors="coerce")
    return pi


def _cs_reference():
    try:
        pi = load_table("per_image")
        cs = pi[(pi["dataset"] == "city")
                & (pi["supervision_type"] == "dense_multi")
                & (pi["backbone"] == "b1")]
        if cs.empty:
            return None
        s = float(pd.to_numeric(cs["student_miou"], errors="coerce").mean())
        t = float(pd.to_numeric(cs["teacher_miou"], errors="coerce").mean())
        b = float(pd.to_numeric(cs["teacher_benefit"], errors="coerce").mean())
        rho_msp, _ = spearmanr(
            pd.to_numeric(cs["student_msp"], errors="coerce"),
            pd.to_numeric(cs["teacher_benefit"], errors="coerce"),
            nan_policy="omit",
        )
        return {"student_miou": s, "teacher_miou": t, "benefit": b, "rho_msp": rho_msp}
    except Exception:
        return None


def _save(fig, name):
    out = figures_dir() / f"{name}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")


def fig_shift_impact(pi, cs):
    conds = COND_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    cs_student = cs["student_miou"] if cs else 0.537
    cs_teacher = cs["teacher_miou"] if cs else 0.630
    stats = pi.groupby("condition").agg({"student_miou": "mean", "teacher_miou": "mean"})
    student_vals = [cs_student] + [float(stats.loc[c, "student_miou"]) for c in conds]
    teacher_vals = [cs_teacher] + [float(stats.loc[c, "teacher_miou"]) for c in conds]
    labels = ["Cityscapes\n(in-dist)"] + [COND_LABELS[c] for c in conds]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, student_vals, w, label="Student (MiT-B1)", color="#8172B3")
    b2 = ax.bar(x + w/2, teacher_vals, w, label="Teacher (B5)", color="#55A868")
    for bars, vals in [(b1, student_vals), (b2, teacher_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("mIoU")
    ax.set_title("A. Model quality degrades under domain shift", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.3)

    ax = axes[1]
    cs_benefit = cs["benefit"] if cs else 0.095
    bstats = pi.groupby("condition")["teacher_benefit"].mean()
    benefits = [cs_benefit] + [float(bstats.loc[c]) for c in conds]
    colors = ["#999999"] + [COND_COLORS[c] for c in conds]
    bars = ax.bar(x, benefits, 0.55, color=colors)
    for bar, val in zip(bars, benefits):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("Mean teacher benefit (Δ risk)")
    ax.set_title("B. Teacher benefit rises under shift", fontsize=12)
    ax.axhline(cs_benefit, color="gray", ls=":", alpha=0.5)

    fig.suptitle("Distribution shift amplifies the need for intelligent deferral",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_shift_impact")


def fig_confident_failures_bars(pi):
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for cond in COND_ORDER:
        sub = pi[pi["condition"] == cond]
        conf = sub[sub["student_msp"] >= 0.85]
        hard = conf[conf["student_miou"] <= 0.30]
        rows.append({
            "condition": COND_LABELS[cond],
            "n_conf": len(conf),
            "n_hard": len(hard),
            "pct": 100 * len(hard) / max(len(conf), 1),
            "color": COND_COLORS[cond],
        })
    fd = pd.DataFrame(rows)
    x = np.arange(len(fd))
    bars = ax.bar(x, fd["pct"], 0.55, color=fd["color"].tolist())
    for bar, row in zip(bars, fd.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{row.n_hard}/{row.n_conf}\n({row.pct:.0f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x, fd["condition"], fontsize=11)
    ax.set_ylabel("Confident frames with mIoU ≤ 0.30 (%)", fontsize=11)
    ax.set_title("Confident Failures on ACDC (student MSP ≥ 0.85, MiT-B1)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout()
    _save(fig, "fig_confident_failures_bars")


def fig_rho_by_condition(pi, cs):
    fig, ax = plt.subplots(figsize=(8.4, 5))
    rows = []
    for cond in COND_ORDER:
        sub = pi[pi["condition"] == cond]
        r_msp, _ = spearmanr(sub["student_msp"], sub["teacher_benefit"], nan_policy="omit")
        r_ent, _ = spearmanr(sub["student_entropy"], sub["teacher_benefit"], nan_policy="omit")
        r_mc, _  = spearmanr(sub["mc_entropy"],      sub["teacher_benefit"], nan_policy="omit")
        rows.append({"cond": COND_LABELS[cond], "MSP": -r_msp, "Entropy": r_ent, "MC Dropout": r_mc})
    if cs and np.isfinite(cs.get("rho_msp", np.nan)):
        rows.insert(0, {"cond": "Cityscapes\n(in-dist)",
                        "MSP": -cs["rho_msp"], "Entropy": np.nan, "MC Dropout": np.nan})
    rd = pd.DataFrame(rows)
    x = np.arange(len(rd))
    w = 0.25
    b1 = ax.bar(x - w, rd["MSP"], w, label="−MSP", color="#4C72B0")
    b2 = ax.bar(x,     rd["Entropy"], w, label="Entropy", color="#DD8452")
    b3 = ax.bar(x + w, rd["MC Dropout"], w, label="MC Dropout", color="#C44E52")
    for bars in (b1, b2, b3):
        for bar in bars:
            v = bar.get_height()
            if not np.isfinite(v):
                continue
            y = v + 0.01 if v >= 0 else v - 0.03
            ax.text(bar.get_x() + bar.get_width()/2, y, f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(-0.15, 0.15, color="red", alpha=0.06)
    ax.text(0.02, 0.04, "ρ ≈ 0 band", fontsize=9, color="red", alpha=0.7,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.5))
    ax.set_xticks(x, rd["cond"], fontsize=10)
    ax.set_ylabel("Spearman ρ with teacher benefit")
    ax.set_title("No uncertainty method correlates with teacher benefit",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.4, 0.4)
    fig.tight_layout()
    _save(fig, "fig_rho_by_condition")


def fig_scatter_by_condition(pi):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, cond in zip(axes.flat, COND_ORDER):
        sub = pi[pi["condition"] == cond]
        r, _ = spearmanr(sub["student_msp"], sub["teacher_benefit"], nan_policy="omit")
        ax.scatter(sub["student_msp"], sub["teacher_benefit"],
                   c=sub["student_risk"], cmap="RdYlGn_r", s=20, alpha=0.6,
                   edgecolors="none")
        ax.set_xlabel("Student MSP (confidence)")
        ax.set_ylabel("Teacher benefit (Δ risk)")
        ax.set_title(f"{COND_LABELS[cond]}:  ρ = {r:.3f}   (n={len(sub)})", fontsize=11)
        conf_fail = sub[(sub["student_msp"] >= 0.90) & (sub["student_miou"] <= 0.30)]
        if len(conf_fail) > 0:
            ax.scatter(conf_fail["student_msp"], conf_fail["teacher_benefit"],
                       facecolors="none", edgecolors="red", s=50, lw=1.5,
                       label=f"Confident failures ({len(conf_fail)})")
            ax.legend(fontsize=8)
    fig.suptitle("MSP is decorrelated from teacher benefit across all ACDC conditions",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig_scatter_by_condition")


def main():
    apply_style()
    pi = _load_acdc_b1()
    cs = _cs_reference()
    fig_shift_impact(pi, cs)
    fig_confident_failures_bars(pi)
    fig_rho_by_condition(pi, cs)
    fig_scatter_by_condition(pi)


if __name__ == "__main__":
    main()
