"""Threshold sweep: CF-AUROC across MSP thresholds.

Shows how Teacher, GT, MaxLogit, and MSP degrade as the confidence bar rises.
One panel per OOD dataset + Cityscapes. The teacher guardrail degrades more
slowly than GT and post-hoc methods under strict thresholds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _lib import (
    apply_style, load_table, savefig, cf_auroc, cf_auroc_stratified,
    pool_acdc_domain, available_datasets, DATASET_LABELS, per_seed_apply,
    ANALYSIS_ROOT,
)


# The sweep stops at 0.95, and a threshold is plotted only where the dataset
# still has this many confident images (Hanley-McNeil standard error 0.06).
# It bites on ACDC alone, which holds 221 images at 0.94 but drops to 144 at
# 0.95; without the floor that last point shows an upturn that is resampling
# noise, not recovery. Every other panel clears the floor at 0.95.
MIN_CONFIDENT = 150

# Panel titles: bare dataset names.
PANEL_TITLES = {
    "city": "Cityscapes", "acdc": "ACDC", "idd": "IDD", "bdd": "BDD100K",
}


def _scorer(col, hi, ds, thr):
    """ACDC uses a per-condition failure cutoff, matching Table 3 and the
    metric definition in the paper. Pooled on ACDC puts almost every confident
    failure in night, so the AUROC degenerates into a night classifier. IDD and
    BDD carry no condition labels and always use the pooled cutoff."""
    def score(g):
        if int((g["student_msp"] >= thr).sum()) < MIN_CONFIDENT:
            return float("nan")
        if ds == "acdc":
            return cf_auroc_stratified(g, col, thr, group_col="condition",
                                       higher_is_fail=hi)
        return cf_auroc(g, col, thr, higher_is_fail=hi)
    return score

# Deep Ensemble lives in its own eval campaign; combined_all has no
# ensemble_entropy column. One three-member ensemble per dataset (no seeds),
# and there is no Cityscapes ensemble, so that panel omits the curve.
DEEP_ENSEMBLE_FILES = {
    "acdc": "b1_per_image.csv",
    "idd":  "iddfull_b1_per_image.csv",
    "bdd":  "bddfull_b1_per_image.csv",
}

# Which score to use for each supervision type
MODE_COL = {
    "dense_multi": ("guardrailpp_utility_dense_gap", True),
    "gt_disagree": ("guardrailpp_utility_dense_bce", True),
    "gt_risk":     ("guardrailpp_utility_dense_gap", True),
}

# Display config: (label, color, ls, lw, marker, ms)
METHOD_STYLE = {
    "T-Multi (ours)": ("#7B68AD", "-",  2.5, "o", 6),
    "GT-Dis":         ("#55A868", "-",  1.8, "D", 4),
    "GT-Gap":         ("#CCB974", "-",  1.6, "v", 4),
    "MaxLogit":       ("#64B5CD", "--", 1.8, "s", 4),
    "MC-Dropout":     ("#C44E52", "--", 1.4, "X", 4),
    "DeepEns":        ("#937860", "--", 1.6, "P", 4),
    "MSP":            ("#aaaaaa", "--", 1.4, "^", 4),
}

# Swept on a 0.01 grid; only the original subset is labelled on the x-axis so
# the ticks stay readable.
THRESHOLDS = tuple(round(0.85 + 0.01 * i, 2) for i in range(11))
# Labels every 0.05, plus the strictest threshold. Gridlines stay at 0.01
# (drawn as minor ticks) so the sweep resolution is visible.
XTICKS = (0.85, 0.90, 0.95)
# Markers only at the labelled thresholds; the line itself is drawn at 0.01.
MARK_AT = [THRESHOLDS.index(t) for t in XTICKS]


def _panel_cap(sub):
    """Strictest threshold at which this dataset still clears MIN_CONFIDENT."""
    ok = [t for t in THRESHOLDS
          if int((sub["student_msp"] >= t).sum()) >= MIN_CONFIDENT]
    return ok[-1] if ok else THRESHOLDS[0]


def _panel_ticks(cap):
    """Labels every 0.05 up to this panel's cap, plus the cap itself."""
    ticks = [t for t in (0.85, 0.90, 0.95) if t <= cap + 1e-9]
    if abs(ticks[-1] - cap) > 1e-9:
        ticks.append(cap)
    return ticks, [THRESHOLDS.index(round(t, 2)) for t in ticks]


def main():
    apply_style()
    pi = load_table("per_image")
    pi = pi[pi["backbone"] == "b1"].copy()
    pi = pool_acdc_domain(pi)

    datasets = available_datasets(pi)
    caps = {ds: _panel_cap(pi[pi["dataset"] == ds]
                           .drop_duplicates(subset=["image_id"]))
            for ds in datasets}
    n_ds = len(datasets)
    ncols = 2
    nrows = (n_ds + 1) // 2
    fig, axes_2d = plt.subplots(nrows, ncols, figsize=(10, 7.5), sharey=False)
    axes = axes_2d.flatten()

    # Per-panel data span, so each axis can be tightened to its own curves
    # instead of sharing one range wide enough for every dataset.
    yspan = {}

    def _draw(ax, label, means, stds, mark=MARK_AT):
        color, ls, lw, marker, ms = METHOD_STYLE[label]
        means = np.asarray(means, dtype=float); stds = np.asarray(stds, dtype=float)
        if np.isfinite(means).any():
            lo = float(np.nanmin(means - stds)); hi = float(np.nanmax(means + stds))
            prev = yspan.get(id(ax))
            yspan[id(ax)] = (min(prev[0], lo), max(prev[1], hi)) if prev else (lo, hi)
        ax.plot(THRESHOLDS, means, color=color, ls=ls, lw=lw, marker=marker,
                markersize=ms, markevery=mark, label=label)
        if (stds > 0).any():
            ax.fill_between(THRESHOLDS, means - stds, means + stds,
                            color=color, alpha=0.18, lw=0)

    def _plot(ax, label, sub, col, hi, ds):
        means, stds = [], []
        for thr in THRESHOLDS:
            m, s, _ = per_seed_apply(sub, _scorer(col, hi, ds, thr))
            means.append(m); stds.append(s if np.isfinite(s) else 0.0)
        _draw(ax, label, means, stds, _panel_ticks(caps[ds])[1])

    def _plot_deep_ensemble(ax, ds):
        """Single 3-member ensemble, so there is no seed spread to shade."""
        fn = DEEP_ENSEMBLE_FILES.get(ds)
        if fn is None or not (ANALYSIS_ROOT / "deep_ensemble_eval" / fn).is_file():
            return
        d = pd.read_csv(ANALYSIS_ROOT / "deep_ensemble_eval" / fn, low_memory=False)
        if ds == "acdc" and "domain" in d.columns:
            d = d[d["domain"] == "all"]
        means = [_scorer("ensemble_entropy", True, ds, thr)(d)
                 for thr in THRESHOLDS]
        _draw(ax, "DeepEns", means, [0.0] * len(THRESHOLDS), _panel_ticks(caps[ds])[1])

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_data = pi[pi["dataset"] == ds]

        # Post-hoc baselines: pull from dense_multi rows so we get all 3 seeds.
        sub_any = ds_data[ds_data["supervision_type"] == "dense_multi"]
        if sub_any.empty:
            sub_any = ds_data.drop_duplicates(subset=["image_id"])

        _plot(ax, "MSP", sub_any, "student_msp", False, ds)
        _plot(ax, "MaxLogit", sub_any, "max_logit", False, ds)
        _plot(ax, "MC-Dropout", sub_any, "mc_entropy", True, ds)
        _plot_deep_ensemble(ax, ds)

        mode_labels = {
            "gt_disagree": "GT-Dis",
            "gt_risk": "GT-Gap",
            "dense_multi": "T-Multi (ours)",
        }
        for mode in ("gt_disagree", "gt_risk", "dense_multi"):
            col, hi = MODE_COL[mode]
            sub = ds_data[ds_data["supervision_type"] == mode]
            if sub.empty or col not in sub.columns:
                continue
            _plot(ax, mode_labels[mode], sub, col, hi, ds)

        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.4)
        ax.set_xlabel("MSP threshold")
        ax.set_title(PANEL_TITLES.get(ds, ds), fontsize=10)
        cap = caps[ds]
        ax.set_xlim(0.845, cap + 0.005)
        lo, hi = yspan.get(id(ax), (0.3, 1.0))
        pad = max(0.02, 0.06 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        panel_ticks = _panel_ticks(cap)[0]
        ax.set_xticks(panel_ticks)
        ax.set_xticklabels([f"{t:.2f}" for t in panel_ticks], fontsize=7.5)
        ax.set_xticks([t for t in THRESHOLDS if t <= cap + 1e-9], minor=True)
        # Vertical grid only: one line per swept threshold, no horizontal rules.
        ax.grid(False)
        ax.grid(True, axis="x", which="major", alpha=0.18, ls="-", lw=0.6)
        ax.grid(True, axis="x", which="minor", alpha=0.18, ls="-", lw=0.5)
        ax.tick_params(axis="x", which="minor", length=0)

    for ax in axes[:n_ds]:
        ax.set_ylabel("CF-AUROC")

    # Hide unused axes
    for i in range(n_ds, len(axes)):
        axes[i].set_visible(False)

    # Shared legend collected across panels: DeepEns is absent from the
    # Cityscapes panel, so the first panel alone would drop it.
    seen = {}
    for ax in axes[:n_ds]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    order = [l for l in ("MSP", "MaxLogit", "MC-Dropout", "DeepEns",
                         "GT-Dis", "GT-Gap", "T-Multi (ours)") if l in seen]
    handles = [seen[l] for l in order]
    labels = order
    fig.legend(handles, labels, ncols=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, -0.01), frameon=False,
               columnspacing=1.5, handlelength=2.2, fontsize=8.5)

    plt.subplots_adjust(bottom=0.12, top=0.94, wspace=0.22, hspace=0.34,
                        left=0.08, right=0.98)
    savefig(fig, "fig_threshold_sweep")


if __name__ == "__main__":
    main()
