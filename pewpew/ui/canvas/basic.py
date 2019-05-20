from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from typing import Tuple


class BasicCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        figsize: Tuple[float, float] = (5.0, 5.0),
        parent: QtWidgets.QWidget = None,
    ):
        fig = Figure(frameon=False, tight_layout=True, figsize=figsize, dpi=100)
        super().__init__(fig)

        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(250, 250)
