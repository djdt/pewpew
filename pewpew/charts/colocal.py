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
        tx: float,
        ty: float,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QPen | None = None,
    ) -> None:
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.red)
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        below = np.logical_or(x < tx, y < ty)
        below_brush = QtGui.QBrush(brush.color().lighter())

        brush = np.where(below, below_brush, brush)

        points = pyqtgraph.ScatterPlotItem(
            x, y, symbol="o", size=5, pen=pen, brush=brush
        )
        self.plot.addItem(points)

    def drawLine(
        self, a: float, b: float, pen: QtGui.QPen | None = None
    ) -> None:
        """Plot a line with gradient 'a' and intercept 'b'."""
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.blue, 1.0)
            pen.setCosmetic(True)

        line = pyqtgraph.PlotCurveItem([0.0, 1.0], [b, 1.0 * a + b], pen=pen)
        self.plot.addItem(line)

    def drawThresholds(
        self, t1: float, t2: float, pen: QtGui.QPen | None = None
    ) -> None:
        """Draw horizontal and vertical lines at 't1' and 't2'."""
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.blue, 1.0, QtCore.Qt.PenStyle.DashLine)
            pen.setCosmetic(True)

        line = pyqtgraph.PlotCurveItem([t1, t1], [0.0, 1.0], pen=pen)
        self.plot.addItem(line)
        line = pyqtgraph.PlotCurveItem([0.0, 1.0], [t2, t2], pen=pen)
        self.plot.addItem(line)
