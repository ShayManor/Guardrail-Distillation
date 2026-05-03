"""ACDC-fog strong figure: a 2x2 panel telling the silent confident
failure story. Crops panels from fig_silent_confident_failure.png so it
runs locally without GPU/checkpoints.

All three heatmaps are placed on the same magma scale where
"bright = high predicted failure". The MSP panel is re-encoded as
(1 - MSP) so it shares semantics with the trained head outputs.

Layout (2 rows x 2 cols):
    Student error | 1 - MSP
    T-Multi (ours) | GT-Dis
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "figures" / "fig_silent_confident_failure.png"
OUT = Path(__file__).resolve().parents[1] / "figures" / "fig_acdc_fog_strong.png"

ROW_Y = (259, 603)  # ACDC fog row
COL_X = {
    "input":    (280, 624),
    "gt":       (639, 983),
    "pred":     (997, 1341),
    "error":    (1356, 1700),
    "msp":      (1714, 2058),
    "teacher":  (2073, 2417),
    "gt_head":  (2431, 2775),
}

UNIFIED_CMAP = "magma"

# (key, title, src_cmap, invert_after_decode)
#   src_cmap=None        → leave the cropped RGB alone (input/error overlay,
#                          and the head panels which are already in magma)
#   src_cmap="viridis"   → decode from viridis scalars and re-encode in magma
#   invert_after_decode  → for MSP we plot (1 - MSP) so bright = failure
PANELS = [
    [("error",   "Student error",   None,      False),
     ("msp",     "1 - MSP",         "viridis", True)],
    [("teacher", "T-Multi (ours)",  None,      False),
     ("gt_head", "GT-Dis",          None,      False)],
]


def _decode_cmap(rgb: np.ndarray, cmap_name: str, n: int = 256) -> np.ndarray:
    """Invert a matplotlib colormap: map RGB pixels back to scalar in [0,1]."""
    lut = (mpl.colormaps[cmap_name](np.linspace(0, 1, n))[:, :3] * 255.0).astype(np.int32)
    flat = rgb.reshape(-1, 3).astype(np.int32)
    # nearest-neighbour LUT lookup (int32 avoids overflow when squaring)
    diffs = ((lut[None, :, :] - flat[:, None, :]) ** 2).sum(axis=2)
    idx = diffs.argmin(axis=1)
    return (idx / (n - 1)).reshape(rgb.shape[:2])


def _recolor(rgb: np.ndarray, src_cmap: str, dst_cmap: str, invert: bool) -> np.ndarray:
    """Decode a baked colormap image and re-encode it in dst_cmap."""
    scalar = _decode_cmap(rgb, src_cmap)
    if invert:
        scalar = 1.0 - scalar
    out = (mpl.colormaps[dst_cmap](scalar)[..., :3] * 255.0).astype(np.uint8)
    return out


def main() -> None:
    src = np.asarray(Image.open(SRC).convert("RGB"))
    y0, y1 = ROW_Y

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.6))
    for r, row in enumerate(PANELS):
        for c, (key, label, src_cmap, invert) in enumerate(row):
            x0, x1 = COL_X[key]
            panel = src[y0:y1, x0:x1]
            if src_cmap is not None:
                panel = _recolor(panel, src_cmap, UNIFIED_CMAP, invert)
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(label, fontsize=14, pad=6)

    # Bold purple frame on our method's panel.
    ours = axes[1, 0]
    for s in ours.spines.values():
        s.set_visible(True)
        s.set_edgecolor("#7B68AD")
        s.set_linewidth(3.0)
    ours.title.set_color("#7B68AD")
    ours.title.set_fontweight("bold")

    # Single shared horizontal colorbar at the bottom for all three heatmaps.
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=UNIFIED_CMAP)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.18, 0.04, 0.64, 0.018])
    cb = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, 1])
    cb.ax.set_xticklabels(["0  (reliable)", "1  (failure)"], fontsize=10)
    cb.ax.tick_params(length=0, pad=2)
    cbar_ax.set_title("predicted failure score", fontsize=10, pad=4)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.10,
                        wspace=0.06, hspace=0.12)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → saved {OUT}")


if __name__ == "__main__":
    main()
