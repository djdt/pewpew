from PySide2 import QtCore, QtWidgets
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

    def copyToClipboard(self) -> None:
        bbox = (
            self.figure.get_tightbbox(
                self.get_renderer(),
            )
            .transformed(self.figure.dpi_scale_trans)
            .padded(5)  # Pad to look nicer
        )
        (x0, y0), (x1, y1) = bbox.get_points().astype(int)
        ymax = self.size().height()  # We need to invert for mpl to Qt
        QtWidgets.QApplication.clipboard().setPixmap(
            self.grab(QtCore.QRect(x0, ymax - y1, x1 - x0, y1 - y0))
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(250, 250)
