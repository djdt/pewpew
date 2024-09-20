import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui

from pewpew.charts.base import SinglePlotGraphicsView


class ColocalisationView(SinglePlotGraphicsView):
    def __init__(
        self,
        parent: pyqtgraph.GraphicsWidget | None = None,
    ):
        super().__init__("Calibration", "Concentration", "Response", parent=parent)
        self.setMinimumSize(320, 320)

        # self.plot.setMouseEnabled(x=False, y=False)
        self.plot.enableAutoRange(x=True, y=True)
        self.plot.setLimits(xMin=-0.05, xMax=1.05, yMin=-0.05, yMax=1.05)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(640, 640)

    def drawPoints(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QPen | None = None,
    ) -> None:
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.red)
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        points = pyqtgraph.ScatterPlotItem(
            x, y, symbol="o", size=10, pen=pen, brush=brush
        )
        self.plot.addItem(points)

    def drawLine(self, a: float, b: float, pen: QtGui.QPen | None = None) -> None:
        """Plot a line with gradient 'a' and intercept 'b'."""
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.red, 1.0)
            pen.setCosmetic(True)

        line = pyqtgraph.PlotCurveItem([b, 1.0], [0.0, a + b], pen=pen)
        self.plot.addItem(line)

    def drawThresholds(
        self, t1: float, t2: float, pen: QtGui.QPen | None = None
    ) -> None:
        """Draw horizontal and vertical lines at 't1' and 't2'."""
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0, QtCore.Qt.PenStyle.DashLine)
            pen.setCosmetic(True)

        line = pyqtgraph.PlotCurveItem([t1, t1], [0.0, 1.0], pen=pen)
        self.plot.addItem(line)
        line = pyqtgraph.PlotCurveItem([0.0, 1.0], [t2, t2], pen=pen)
        self.plot.addItem(line)
