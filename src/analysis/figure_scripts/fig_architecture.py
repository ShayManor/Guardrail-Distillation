"""Four separate method diagrams rendered into one PNG.

1. architecture — frozen student, guardrail head, two dense outputs.
2. Stage-4 training — targets from the teacher or from the labels.
3. test time — severity map to an accept/defer decision.
4. fusion — the guardrail severity and the post-hoc energy score rank-averaged
   into fusion_guard_energy, the headline selective score.

Each diagram is self-contained and gets its own caption in the paper, so none
of them carries a title here. The input and the dense maps are real: one ACDC
fog val frame through the mit-b1 SKD student and the dense_multi guardrail
head, precomputed into assets/arch_panels.npz by scripts/dump_arch_panels.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from _lib import apply_style, savefig
import matplotlib.pyplot as plt

BLK = "#1a1a1a"
GREY_FILL = "#c9c9c9"
CHIP_FILL = "#dbe5f1"
ORANGE = "#e9a13b"
ENC_BAR = "#4a86c8"
DEC_BAR = "#e8912d"
GUARD_BAR = "#7c5ca5"
ACCEPT_FILL = "#b6d7a8"
DEFER_FILL = "#ea9999"
C_DIS = "#5c4a86"
C_RISK = "#a8331f"
C_EN = "#1f6f68"
MUTED = "#6f7378"

PANELS = Path(__file__).resolve().parent / "assets" / "arch_panels.npz"

FIGSIZE = (11.5, 15.9)
XLIM = 100.0
YLIMS = (20.0, 44.0, 16.0, 21.0)

CMAP_D = LinearSegmentedColormap.from_list("d", ["#f6f4fa", "#5c4a86"])
CMAP_R = LinearSegmentedColormap.from_list("r", ["#fff6ee", "#a8331f"])
CMAP_E = LinearSegmentedColormap.from_list("e", ["#f1f7f6", "#1f6f68"])


# ----------------------------------------------------------------- primitives
def rect(ax, x0, y0, x1, y1, *, fc="white", ec=BLK, lw=1.2, z=2):
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc,
                                    edgecolor=ec, linewidth=lw, zorder=z))


def chip(ax, cx, cy, text, *, w=7.0, h=3.4, fc=CHIP_FILL, size=10.5):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.45", facecolor=fc,
        edgecolor=BLK, linewidth=1.0, zorder=3))
    ax.text(cx, cy, text, fontsize=size, color=BLK, ha="center", va="center",
            zorder=8)


def module(ax, x_left, cy, layers, *, bw=2.2, gap=1.2, bh=9.6, pad=0.45,
           fc="white", lab=8.0):
    """Module box hugging its layer slabs. layers is [(colour, name), ...]."""
    inner = len(layers) * bw + (len(layers) - 1) * gap
    x_right = x_left + inner + 2 * pad
    rect(ax, x_left, cy - bh / 2 - pad, x_right, cy + bh / 2 + pad, fc=fc)
    x = x_left + pad
    for color, name in layers:
        ax.add_patch(mpatches.Rectangle(
            (x, cy - bh / 2), bw, bh, facecolor=color, edgecolor=BLK,
            linewidth=0.8, zorder=4))
        ax.text(x + bw / 2, cy, name, fontsize=lab, color="white",
                ha="center", va="center", rotation=90, zorder=8)
        x += bw + gap
    return x_right


def arrow(ax, p0, p1, *, color=BLK, lw=1.1, z=5, ms=9):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, linewidth=lw,
        zorder=z, shrinkA=0, shrinkB=0))


def elbow(ax, pts, *, color=BLK, lw=1.1, z=5, ms=9):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round",
            solid_joinstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, z=z, ms=ms)


def txt(ax, x, y, s, *, size=10, weight="normal", color=BLK, ha="center"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha,
            va="center", zorder=8)


def img(ax, arr, x0, y0, x1, y1, *, cmap=None, lw=1.1, ec=BLK):
    kw = {}
    if cmap is not None:
        # nan-aware: the dense targets carry NaN at ignored pixels.
        lo, hi = np.nanpercentile(arr, (50, 98))
        kw = dict(vmin=lo, vmax=hi)
    ax.imshow(arr, extent=[x0, x1, y0, y1], aspect="auto", cmap=cmap, zorder=3,
              interpolation="bilinear", **kw)
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                                    edgecolor=ec, lw=lw, zorder=5))


def make_square(fig, ax, ylim):
    bb = ax.get_position()
    w_in = bb.width * fig.get_size_inches()[0]
    h_in = bb.height * fig.get_size_inches()[1]
    ratio = (h_in / ylim) / (w_in / XLIM)
    return lambda cx, h: (cx - h * ratio / 2, cx + h * ratio / 2)


def new_axes(fig, gs, ylim):
    ax = fig.add_subplot(gs)
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, ylim)
    ax.axis("off")
    return ax, make_square(fig, ax, ylim)


STUDENT_LAYERS = [(ENC_BAR, f"stage {i}") for i in (1, 2, 3, 4)] + \
                 [(DEC_BAR, "decoder")]
GUARD_LAYERS = [(GUARD_BAR, "conv 64"), (GUARD_BAR, "conv 64"),
                (GUARD_BAR, "conv 32")]


# -------------------------------------------------------------- architecture
def draw_architecture(ax, sq, data):
    ix0, ix1 = sq(8.0, 9.0)
    img(ax, data["rgb"], ix0, 5.5, ix1, 14.5)

    sx1 = module(ax, 18.0, 10.0, STUDENT_LAYERS)
    txt(ax, (18.0 + sx1) / 2, 16.9, "Student", size=11.5, weight="bold")
    txt(ax, (18.0 + sx1) / 2, 3.0, "SegFormer mit-b0 / b1 / b2", size=8.5,
        color=MUTED)
    arrow(ax, (ix1, 10.0), (18.0, 10.0))

    chip(ax, 42.5, 10.0, "$z^{S}$", w=6.0, h=3.2)
    arrow(ax, (sx1, 10.0), (39.5, 10.0))
    arrow(ax, (45.5, 10.0), (50.0, 10.0))

    gx1 = module(ax, 50.0, 10.0, GUARD_LAYERS, bw=2.8, gap=1.6)
    txt(ax, (50.0 + gx1) / 2, 16.9, "Guardrail", size=11.5, weight="bold")
    txt(ax, (50.0 + gx1) / 2, 3.0, "+0.36M params", size=8.5, color=MUTED)

    mx0, mx1 = sq(80.0, 6.0)
    img(ax, data["d"], mx0, 10.6, mx1, 16.6, cmap=CMAP_D, ec=C_DIS, lw=1.3)
    txt(ax, (mx0 + mx1) / 2, 17.9, "disagreement", size=10, weight="bold",
        color=C_DIS)
    img(ax, data["r"], mx0, 2.4, mx1, 8.4, cmap=CMAP_R, ec=C_RISK, lw=1.3)
    txt(ax, (mx0 + mx1) / 2, 1.2, "severity", size=10, weight="bold", color=C_RISK)

    arrow(ax, (gx1, 11.6), (mx0, 13.6))
    arrow(ax, (gx1, 8.4), (mx0, 5.4))


# ------------------------------------------------------------------- training
def draw_training(ax, sq, data):
    ix0, ix1 = sq(6.0, 8.0)
    img(ax, data["rgb"], ix0, 23.0, ix1, 31.0)

    sx1 = module(ax, 14.0, 37.0, STUDENT_LAYERS, bw=2.0, gap=1.0, bh=9.0,
                 pad=0.45, lab=7.0)
    txt(ax, (14.0 + sx1) / 2, 43.2, "Student", size=11, weight="bold")

    rect(ax, 14.0, 23.0, sx1, 31.0, fc=GREY_FILL)
    txt(ax, (14.0 + sx1) / 2, 28.0, "Teacher", size=11, weight="bold")
    txt(ax, (14.0 + sx1) / 2, 25.6, "SegFormer-B5", size=8.5, color="#3d4045")

    chip(ax, (14.0 + sx1) / 2, 9.0, "labels   $y$", w=sx1 - 14.0, h=4.0)

    arrow(ax, (ix1, 27.6), (14.0, 35.0))
    arrow(ax, (ix1, 27.0), (14.0, 27.0))

    chip(ax, 33.5, 37.0, "$z^{S}$", w=5.6, h=3.2)
    chip(ax, 33.5, 27.0, "$z^{T}$", w=5.6, h=3.2)
    arrow(ax, (sx1, 37.0), (30.7, 37.0))
    arrow(ax, (sx1, 27.0), (30.7, 27.0))

    gx1 = module(ax, 41.0, 37.0, GUARD_LAYERS, bw=2.6, gap=1.5, bh=9.0,
                 pad=0.45, lab=7.0)
    txt(ax, (41.0 + gx1) / 2, 43.2, "Guardrail", size=11, weight="bold")
    arrow(ax, (36.3, 37.0), (41.0, 37.0))

    # every target needs the student; the teacher and the labels supply the
    # reference, so all three feed both target blocks
    ax.plot([37.5, 37.5], [37.0, 9.0], color=BLK, lw=1.0, zorder=4,
            solid_capstyle="round")
    ax.plot([36.3, 37.5], [27.0, 27.0], color=BLK, lw=1.0, zorder=4)
    ax.plot([sx1, 37.5], [9.0, 9.0], color=BLK, lw=1.0, zorder=4)
    arrow(ax, (37.5, 25.5), (40.0, 25.5), lw=1.0)
    arrow(ax, (37.5, 8.5), (40.0, 8.5), lw=1.0)

    # Each target row carries the map it produces for this frame, so the
    # supervision the guardrail actually receives is visible, not just named.
    lanes = [
        (32.0, 19.0, "disagreement target", CMAP_D,
         (r"$\mathbf{1}[\,\hat y^{S}_{ij}\neq\hat y^{T}_{ij}\,]$", "d_t"),
         (r"$\mathbf{1}[\,\hat y^{S}_{ij}\neq y_{ij}\,]$", "d_gt")),
        (15.0, 2.0, "severity target", CMAP_R,
         (r"$\mathrm{CE}(z^{S}_{ij},y_{ij})-\mathrm{CE}(z^{T}_{ij},y_{ij})$", "s_t"),
         (r"$\mathrm{CE}(z^{S}_{ij},\,y_{ij})$", "s_gt")),
    ]
    for y1, y0, title, cmap, t_row, l_row in lanes:
        rect(ax, 40.0, y0, 74.0, y1)
        txt(ax, 57.0, y1 - 1.6, title, size=9, weight="bold", color=MUTED)
        for (expr, key), lab, cy in ((t_row, "teacher", y1 - 5.0),
                                     (l_row, "labels", y1 - 9.8)):
            txt(ax, 42.0, cy, lab, size=8.5, color=MUTED, ha="left")
            txt(ax, 56.0, cy, expr, size=9.5)
            if key in data:
                mx0, mx1 = sq(70.0, 4.4)
                img(ax, data[key], mx0, cy - 2.2, mx1, cy + 2.2, cmap=cmap,
                    lw=0.8)

    # each loss takes its target from the left and its prediction from above
    chip(ax, 87.0, 32.0, r"$\hat d$", w=6.0, h=3.0, fc="#e6e1f0")
    chip(ax, 87.0, 15.0, r"$\hat s$", w=6.0, h=3.0, fc="#f7e3dd")
    for key, cy, cmap, ec in (("d", 32.0, CMAP_D, C_DIS), ("r", 15.0, CMAP_R, C_RISK)):
        px0, px1 = sq(92.8, 4.4)
        img(ax, data[key], px0, cy - 2.2, px1, cy + 2.2, cmap=cmap, lw=1.0, ec=ec)
    rect(ax, 79.0, 22.2, 95.0, 28.8, fc=ORANGE)
    txt(ax, 87.0, 25.5, r"$\mathcal{L}_{\mathrm{BCE}}$", size=12)
    rect(ax, 79.0, 5.2, 95.0, 11.8, fc=ORANGE)
    txt(ax, 87.0, 8.5, r"$\mathcal{L}_{\mathrm{smooth}\text{-}\ell_1}$", size=12)

    arrow(ax, (74.0, 25.5), (79.0, 25.5), lw=1.2)
    arrow(ax, (74.0, 8.5), (79.0, 8.5), lw=1.2)
    elbow(ax, [(gx1, 38.4), (87.0, 38.4), (87.0, 33.5)], lw=1.2)
    arrow(ax, (87.0, 30.5), (87.0, 28.8), lw=1.2)
    elbow(ax, [(gx1, 35.6), (75.5, 35.6), (75.5, 15.0), (84.0, 15.0)], lw=1.2)
    arrow(ax, (87.0, 13.5), (87.0, 11.8), lw=1.2)


# --------------------------------------------------------------------- test time
def draw_inference(ax, sq, data):
    ix0, ix1 = sq(6.5, 8.0)
    img(ax, data["rgb"], ix0, 4.0, ix1, 12.0)

    sx1 = module(ax, 15.0, 8.0, STUDENT_LAYERS, bw=2.0, gap=1.0, bh=9.0,
                 pad=0.45, lab=7.0)
    txt(ax, (15.0 + sx1) / 2, 14.3, "Student", size=11, weight="bold")

    gx1 = module(ax, 34.0, 8.0, GUARD_LAYERS, bw=2.4, gap=1.4, bh=9.0, pad=0.45, lab=7.0)
    txt(ax, (34.0 + gx1) / 2, 14.3, "Guardrail", size=11, weight="bold")

    arrow(ax, (ix1, 8.0), (15.0, 8.0))
    arrow(ax, (sx1, 8.0), (34.0, 8.0))

    rx0, rx1 = sq(54.0, 8.0)
    img(ax, data["r"], rx0, 4.0, rx1, 12.0, cmap=CMAP_R, ec=C_RISK, lw=1.3)
    txt(ax, (rx0 + rx1) / 2, 13.3, "severity", size=10, weight="bold", color=C_RISK)
    arrow(ax, (gx1, 8.0), (rx0, 8.0))

    rect(ax, 63.0, 5.6, 74.0, 10.4, fc=GREY_FILL)
    txt(ax, 68.5, 8.0, "average", size=10)
    arrow(ax, (rx1, 8.0), (63.0, 8.0))

    chip(ax, 80.5, 8.0, "$g(x)$", w=7.0, h=3.6)
    arrow(ax, (74.0, 8.0), (77.0, 8.0))

    txt(ax, 86.5, 14.0, r"compare $\tau$", size=9, color=MUTED)
    arrow(ax, (84.0, 8.0), (90.0, 11.0))
    arrow(ax, (84.0, 8.0), (90.0, 5.0))
    chip(ax, 94.5, 11.0, "accept", w=9.0, h=3.2, fc=ACCEPT_FILL, size=10)
    chip(ax, 94.5, 5.0, "defer", w=9.0, h=3.2, fc=DEFER_FILL, size=10)


# ----------------------------------------------------------------------- fusion
def draw_fusion(ax, sq, data):
    ix0, ix1 = sq(5.5, 7.0)
    img(ax, data["rgb"], ix0, 6.5, ix1, 13.5)

    sx1 = module(ax, 13.0, 10.0, STUDENT_LAYERS, bw=1.9, gap=0.95, bh=7.6,
                 pad=0.45, lab=6.5)
    txt(ax, (13.0 + sx1) / 2, 15.1, "Student", size=11, weight="bold")
    arrow(ax, (ix1, 10.0), (13.0, 10.0))

    chip(ax, 31.5, 10.0, "$z^{S}$", w=5.2, h=3.0)
    arrow(ax, (sx1, 10.0), (28.9, 10.0))

    # the same logits feed the learned head and the training-free energy score
    elbow(ax, [(34.1, 10.0), (36.6, 10.0), (36.6, 15.5), (38.5, 15.5)])
    elbow(ax, [(34.1, 10.0), (36.6, 10.0), (36.6, 5.0), (38.5, 5.0)])

    gx1 = module(ax, 38.5, 15.5, GUARD_LAYERS, bw=2.1, gap=1.2, bh=6.2,
                 pad=0.45, lab=6.5)
    txt(ax, (38.5 + gx1) / 2, 20.0, "Guardrail", size=10.5, weight="bold")
    mx0, mx1 = sq(54.5, 6.2)
    img(ax, data["r"], mx0, 12.4, mx1, 18.6, cmap=CMAP_R, ec=C_RISK, lw=1.3)
    txt(ax, (mx0 + mx1) / 2, 20.0, "severity", size=9.5, weight="bold",
        color=C_RISK)
    arrow(ax, (gx1, 15.5), (mx0, 15.5))

    rect(ax, 38.5, 1.9, gx1, 8.1, fc=GREY_FILL)
    txt(ax, (38.5 + gx1) / 2, 5.0, r"$-\log\sum_c e^{z^{S}_{c}}$", size=9.5)
    txt(ax, (38.5 + gx1) / 2, 9.4, "Energy", size=10.5, weight="bold")
    img(ax, data["e"], mx0, 1.9, mx1, 8.1, cmap=CMAP_E, ec=C_EN, lw=1.3)
    txt(ax, (mx0 + mx1) / 2, 0.7, "energy", size=9.5, weight="bold", color=C_EN)
    arrow(ax, (gx1, 5.0), (mx0, 5.0))

    for cy, label in ((15.5, "$g(x)$"), (5.0, "$E(x)$")):
        arrow(ax, (mx1, cy), (61.6, cy))
        txt(ax, (mx1 + 61.6) / 2, cy + 1.4, "mean", size=7.5, color=MUTED)
        chip(ax, 65.0, cy, label, w=6.2, h=3.0)

    elbow(ax, [(68.1, 15.5), (71.0, 15.5), (71.0, 11.6), (73.0, 11.6)])
    elbow(ax, [(68.1, 5.0), (71.0, 5.0), (71.0, 8.4), (73.0, 8.4)])
    rect(ax, 73.0, 6.4, 86.0, 13.6)
    txt(ax, 79.5, 11.4, "rank average", size=10)
    txt(ax, 79.5, 8.6, r"$\mathrm{rk}(g)+\mathrm{rk}(E)$", size=8.5, color=MUTED)

    arrow(ax, (86.0, 10.0), (91.0, 13.0))
    arrow(ax, (86.0, 10.0), (91.0, 7.0))
    chip(ax, 95.5, 13.0, "accept", w=8.5, h=3.0, fc=ACCEPT_FILL, size=10)
    chip(ax, 95.5, 7.0, "defer", w=8.5, h=3.0, fc=DEFER_FILL, size=10)


def main():
    apply_style()
    data = np.load(PANELS)
    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(4, 1, figure=fig, height_ratios=list(YLIMS), hspace=0.26,
                  left=0.02, right=0.98, top=0.98, bottom=0.02)

    ax1, sq1 = new_axes(fig, gs[0], YLIMS[0])
    draw_architecture(ax1, sq1, data)
    ax2, sq2 = new_axes(fig, gs[1], YLIMS[1])
    draw_training(ax2, sq2, data)
    ax3, sq3 = new_axes(fig, gs[2], YLIMS[2])
    draw_inference(ax3, sq3, data)
    ax4, sq4 = new_axes(fig, gs[3], YLIMS[3])
    draw_fusion(ax4, sq4, data)

    for upper, lower in ((ax1, ax2), (ax2, ax3), (ax3, ax4)):
        y = (upper.get_position().y0 + lower.get_position().y1) / 2
        fig.add_artist(Line2D([0.03, 0.97], [y, y], color="#e0e1e4", lw=1.0))

    savefig(fig, "fig_architecture")


if __name__ == "__main__":
    main()
