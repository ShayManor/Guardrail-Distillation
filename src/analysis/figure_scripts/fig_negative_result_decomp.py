"""F10 — Why teacher_benefit is unpredictable.

Left panel shows *why*: student and teacher risk are tightly coupled, so their
difference is a thin residual. Right panel shows the consequence — that residual
is not recoverable from the student's own logits, at R^2 ~ 0 out-of-fold.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _lib import (
    apply_style, load_table, savefig, available_datasets,
    pool_acdc_domain, BACKBONE_LABELS,
)

# Matches fig_supervision_ablation: grey for in-domain, OOD carries the color.
DATASET_COLORS = {
    "city": "#9aa0a6",
    "acdc": "#3b6ea5",
    "idd":  "#d97c2f",
    "bdd":  "#7a4ea0",
}

SHORT_LABELS = {
    "city": "Cityscapes",
    "acdc": "ACDC",
    "idd":  "IDD",
    "bdd":  "BDD100K",
}

SCATTER_BACKBONE = "b1"
# Single seed for the scatters: the CV panel must not see the same image in
# both a train and a test fold, and pooled seeds would triplicate every image.
SCATTER_SEED = 42

# Training-free post-hoc statistics of the student's own logits. Used as the
# feature set for the predictability ceiling in the right-hand panel.
POSTHOC_FEATURES = [
    "student_msp", "student_msp_std", "student_entropy", "student_entropy_std",
    "temp_msp", "temp_entropy", "low_conf_frac_050", "low_conf_frac_070",
    "mc_entropy", "mc_mutual_info", "energy_score", "max_logit",
]


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["supervision_type"] == "dense_multi"].copy()
    pi = pool_acdc_domain(pi)

    datasets = available_datasets(pi)

    fig, (ax_s, ax_p) = plt.subplots(
        1, 2, figsize=(10.6, 4.5),
        gridspec_kw={"wspace": 0.24},
    )

    scat = pi[pi["seed"] == SCATTER_SEED]

    # ---- Left: per-image student vs teacher risk (one backbone) ------------
    allv = []
    fits = []
    corrs = []
    for ds in datasets:
        sub = scat[(scat["dataset"] == ds) & (scat["backbone"] == SCATTER_BACKBONE)]
        sr = pd.to_numeric(sub["student_risk"], errors="coerce")
        tr = pd.to_numeric(sub["teacher_risk"], errors="coerce")
        m = sr.notna() & tr.notna()
        if not m.any():
            continue
        x, y = sr[m].to_numpy(), tr[m].to_numpy()
        corrs.append((ds, float(np.corrcoef(x, y)[0, 1])))
        ax_s.scatter(x, y, s=5, alpha=0.28, linewidths=0,
                     color=DATASET_COLORS.get(ds, "#444"), zorder=3)
        # Least-squares fit, drawn only over the range this dataset covers.
        slope, intercept = np.polyfit(x, y, 1)
        xf = np.array([np.percentile(x, 1), np.percentile(x, 99)])
        fits.append((ds, xf, slope * xf + intercept))
        allv.append(np.concatenate([x, y]))

    for ds, xf, yf in fits:
        ax_s.plot(xf, yf, lw=1.8, color=DATASET_COLORS.get(ds, "#444"),
                  path_effects=[pe.Stroke(linewidth=3.4, foreground="white"),
                                pe.Normal()], zorder=4)

    # Percentile limits so a lone outlier doesn't stretch the panel.
    allv = np.concatenate(allv)
    lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ax_s.set_xlim(lo, hi)
    ax_s.set_ylim(lo, hi)
    ax_s.set_aspect("equal")
    ax_s.grid(False)
    ax_s.set_xlabel("Student risk")
    ax_s.set_ylabel("Teacher risk")
    ax_s.text(0.03, 0.97, BACKBONE_LABELS[SCATTER_BACKBONE],
              transform=ax_s.transAxes, ha="left", va="top",
              fontsize=9, color="#666")
    for i, (ds, rv) in enumerate(corrs):
        ax_s.text(0.03, 0.89 - 0.075 * i,
                  f"{SHORT_LABELS.get(ds, ds)}  r={rv:+.2f}",
                  transform=ax_s.transAxes, ha="left", va="top", fontsize=8.5,
                  color=DATASET_COLORS.get(ds, "#444"))

    # ---- Right: best-case cross-validated prediction of the benefit -------
    # Ridge on training-free post-hoc features, out-of-fold predictions. This
    # is a ceiling argument: it does not depend on how the guardrail trained.
    pfits, r2s = [], []
    px_all, py_all = [], []
    for ds in datasets:
        sub = scat[(scat["dataset"] == ds) & (scat["backbone"] == SCATTER_BACKBONE)]
        cols = [c for c in POSTHOC_FEATURES if c in sub.columns]
        block = (sub[cols + ["student_risk", "teacher_risk"]]
                 .apply(pd.to_numeric, errors="coerce").dropna())
        if len(block) < 100:
            continue
        X = block[cols].to_numpy()
        y = (block["student_risk"] - block["teacher_risk"]).to_numpy()
        model = make_pipeline(StandardScaler(),
                              RidgeCV(alphas=np.logspace(-3, 3, 13)))
        yhat = cross_val_predict(model, X, y,
                                 cv=KFold(5, shuffle=True, random_state=0))
        r2s.append((ds, r2_score(y, yhat)))
        ax_p.scatter(y, yhat, s=5, alpha=0.28, linewidths=0,
                     color=DATASET_COLORS.get(ds, "#444"), zorder=3)
        slope, intercept = np.polyfit(y, yhat, 1)
        xf = np.array([np.percentile(y, 1), np.percentile(y, 99)])
        pfits.append((ds, xf, slope * xf + intercept))
        px_all.append(y)
        py_all.append(yhat)

    for ds, xf, yf in pfits:
        ax_p.plot(xf, yf, lw=1.8, color=DATASET_COLORS.get(ds, "#444"),
                  path_effects=[pe.Stroke(linewidth=3.4, foreground="white"),
                                pe.Normal()], zorder=4)

    px_all = np.concatenate(px_all)
    py_all = np.concatenate(py_all)
    # Shared limits on both axes: a perfect predictor would trace the diagonal,
    # so the collapse of the predictions into a flat band is shown to scale.
    both = np.concatenate([px_all, py_all])
    plo, phi = np.percentile(both, 0.3), np.percentile(both, 99.7)
    ppad = 0.04 * (phi - plo)
    ax_p.set_xlim(plo - ppad, phi + ppad)
    ax_p.set_ylim(plo - ppad, phi + ppad)
    ax_p.set_aspect("equal")
    ax_p.grid(False)
    ax_p.set_xlabel("True benefit")
    ax_p.set_ylabel("Predicted benefit")
    for i, (ds, v) in enumerate(r2s):
        ax_p.text(0.03, 0.97 - 0.075 * i, f"{SHORT_LABELS.get(ds, ds)}  R²={v:+.2f}",
                  transform=ax_p.transAxes, ha="left", va="top", fontsize=8.5,
                  color=DATASET_COLORS.get(ds, "#444"))

    # Every panel labels its own series, so no shared legend is needed.
    plt.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.97)
    savefig(fig, "fig_negative_result_decomp")


if __name__ == "__main__":
    main()
