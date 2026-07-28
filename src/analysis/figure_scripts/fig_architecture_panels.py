"""The four fig_architecture diagrams, each written to its own PNG.

fig_architecture.py stacks them into a single figure; the paper places them
separately with their own captions, so this emits one file per panel using the
same drawing code.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _lib import apply_style, savefig
from fig_architecture import (
    PANELS,
    XLIM,
    YLIMS,
    draw_architecture,
    draw_fusion,
    draw_inference,
    draw_training,
    make_square,
)

WIDTH = 11.5
PANEL_NAMES = ("fig_arch_overview", "fig_arch_training", "fig_arch_inference",
               "fig_arch_fusion")
DRAWERS = (draw_architecture, draw_training, draw_inference, draw_fusion)


def main():
    apply_style()
    data = np.load(PANELS)
    for name, drawer, ylim in zip(PANEL_NAMES, DRAWERS, YLIMS):
        fig = plt.figure(figsize=(WIDTH, WIDTH * ylim / XLIM))
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
        ax.set_xlim(0, XLIM)
        ax.set_ylim(0, ylim)
        ax.axis("off")
        drawer(ax, make_square(fig, ax, ylim), data)
        savefig(fig, name)


if __name__ == "__main__":
    main()
