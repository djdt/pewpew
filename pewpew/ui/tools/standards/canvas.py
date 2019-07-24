from PySide2 import QtWidgets
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pewpew.ui.canvas.laser import LaserCanvas

from typing import List


class StandardsCanvas(LaserCanvas):
    def __init__(self, viewconfig: dict, parent: QtWidgets.QWidget = None):
        options = {"colorbar": False, "scalebar": False, "label": False}
        super().__init__(viewconfig, options=options, parent=parent)
        div = make_axes_locatable(self.ax)
        self.bax = div.append_axes("left", size=0.2, pad=0, sharey=self.ax)
        self.bax.get_xaxis().set_visible(False)
        self.bax.get_yaxis().set_visible(False)
        self.bax.set_facecolor("black")

    def drawLevels(self, texts: List[str], levels: int) -> None:
        self.bax.clear()
        ax_fraction = 1.0 / levels
        # Draw lines
        for frac in np.linspace(1.0 - ax_fraction, ax_fraction, levels - 1):
            line = Line2D(
                (0.0, 1.0),
                (frac, frac),
                transform=self.ax.transAxes,
                color="black",
                linestyle="--",
                linewidth=2.0,
            )
            self.ax.add_artist(line)

        for i, frac in enumerate(np.linspace(1.0, ax_fraction, levels)):
            text = Text(
                x=0.5,
                y=frac - (ax_fraction / 2.0),
                text=texts[i],
                transform=self.bax.transAxes,
                color="white",
                fontsize=12,
                horizontalalignment="center",
                verticalalignment="center",
            )
            self.bax.add_artist(text)
