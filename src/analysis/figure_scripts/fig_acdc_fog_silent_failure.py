"""Strong silent-confident-failure figure: 4-row x 4-col grid.

One row per dataset/domain (ACDC fog, ACDC snow, IDD India, BDD100K US),
four columns per row:
    Student error | 1 - MSP | T-Multi (ours) | GT-Dis

Crops panels from fig_silent_confident_failure.png so this runs locally
without GPU/checkpoints. All heatmaps share a magma scale where
"bright = high predicted failure"; the MSP panel is re-encoded as (1 - MSP)
to share the same semantics. The "T-Multi (ours)" column is highlighted
with a purple frame.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "figures" / "fig_silent_confident_failure.png"
OUT = Path(__file__).resolve().parents[1] / "figures" / "fig_acdc_fog_strong.png"

# Row Y-ranges in the source grid (auto-detected from GT column).
# ACDC images are 2:1 aspect; IDD/BDD are 4:1 — row heights differ.
ROWS = [
    ("ACDC fog",     (259, 603)),
    ("ACDC snow",    (617, 960)),
    ("IDD (India)",  (1060, 1232)),
    ("BDD100K (US)", (1418, 1590)),
]

COL_X = {
    "error":   (1356, 1700),
    "msp":     (1714, 2058),
    "teacher": (2073, 2417),
    "gt_head": (2431, 2775),
}

# (key, title, src_cmap, invert_after_decode)
#   src_cmap=None        → leave the cropped RGB alone
#   src_cmap="viridis"   → decode from viridis scalars and re-encode in magma
#   invert_after_decode  → for MSP we plot (1 - MSP) so bright = failure
COLS = [
    ("error",   "Student error",  None,      False),
    ("msp",     "1 - MSP",        "viridis", True),
    ("gt_head", "GT-Dis",         None,      False),
    ("teacher", "T-Multi (ours)", None,      False),
]

UNIFIED_CMAP = "magma"
OURS_COLOR = "#7B68AD"


def _decode_cmap(rgb: np.ndarray, cmap_name: str, n: int = 256) -> np.ndarray:
    """Invert a matplotlib colormap: map RGB pixels back to scalar in [0,1]."""
    lut = (mpl.colormaps[cmap_name](np.linspace(0, 1, n))[:, :3] * 255.0).astype(np.int32)
    flat = rgb.reshape(-1, 3).astype(np.int32)
    diffs = ((lut[None, :, :] - flat[:, None, :]) ** 2).sum(axis=2)
    idx = diffs.argmin(axis=1)
    return (idx / (n - 1)).reshape(rgb.shape[:2])


def _recolor(rgb: np.ndarray, src_cmap: str, dst_cmap: str, invert: bool) -> np.ndarray:
    scalar = _decode_cmap(rgb, src_cmap)
    if invert:
        scalar = 1.0 - scalar
    return (mpl.colormaps[dst_cmap](scalar)[..., :3] * 255.0).astype(np.uint8)


def main() -> None:
    src = np.asarray(Image.open(SRC).convert("RGB"))

    n_rows, n_cols = len(ROWS), len(COLS)
    # Column width in source pixels (all COL_X spans are equal width).
    col_w = (COL_X["error"][1] - COL_X["error"][0])
    # height_ratios = native panel height / panel width; this matches what
    # imshow(aspect="equal") will actually render, so cells contain no slack.
    height_ratios = [(y1 - y0) / col_w for _, (y0, y1) in ROWS]

    fig_w = 2.4 * n_cols
    panel_w_in = (fig_w * 0.92) / n_cols
    fig_h = panel_w_in * sum(height_ratios) + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows, n_cols, figure=fig,
        height_ratios=height_ratios,
        wspace=0.04, hspace=0.08,
        left=0.06, right=0.99, top=0.96, bottom=0.10,
    )

    ours_col = next(i for i, (k, *_rest) in enumerate(COLS) if k == "teacher")

    for r, (row_label, (y0, y1)) in enumerate(ROWS):
        for c, (key, title, src_cmap, invert) in enumerate(COLS):
            x0, x1 = COL_X[key]
            panel = src[y0:y1, x0:x1]
            if src_cmap is not None:
                panel = _recolor(panel, src_cmap, UNIFIED_CMAP, invert)
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=12, pad=4,
                             color=OURS_COLOR if c == ours_col else "black",
                             fontweight="bold" if c == ours_col else "normal")
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11, rotation=0,
                              ha="right", va="center", labelpad=14)
            if c == ours_col:
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_edgecolor(OURS_COLOR)
                    s.set_linewidth(2.0)

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=UNIFIED_CMAP)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.045, 0.50, 0.012])
    cb = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, 1])
    cb.ax.tick_params(length=0, pad=2, labelsize=10)
    cbar_ax.set_xlabel("predicted failure score", fontsize=10, labelpad=4)

    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → saved {OUT}")


if __name__ == "__main__":
    main()
