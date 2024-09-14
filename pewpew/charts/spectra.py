import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui

from pewpew.charts.base import SinglePlotGraphicsView, ViewBoxForceScaleAtZero


class SpectraView(SinglePlotGraphicsView):
    def __init__(
        self,
        parent: pyqtgraph.GraphicsWidget | None = None,
    ):
        super().__init__(
            "Mass Spectra",
            "m/z",
            "Response",
            viewbox=ViewBoxForceScaleAtZero(),
            parent=parent,
        )
        self.setMinimumSize(320, 160)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(640, 240)

    def drawCentroidSpectra(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        xs = np.repeat(x, 2)
        ys = np.stack((np.zeros_like(y), y), axis=1).flat
        line = pyqtgraph.PlotCurveItem(
            xs, ys, pen=pen, connect="pairs", skipFiniteCheck=True
        )
        self.plot.addItem(line)
