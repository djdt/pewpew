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

        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.enableAutoRange(x=True, y=True)

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

        if self.points is not None:
            self.plot.removeItem(self.points)

        points = pyqtgraph.ScatterPlotItem(
            x, y, symbol="o", size=10, pen=pen, brush=brush
        )
        self.plot.addItem(points)

    def drawLine(self, a: float, b: float, pen: QtGui.QPen | None = None) -> None:
        """Plot a line with gradient 'a' and intercept 'b'."""
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.red, 1.0)
            pen.setCosmetic(True)

        line = pyqtgraph.PlotCurveItem([b, 0.0], [1.0, a + b], pen=pen)
        self.plot.addItem(line)

    def drawThresholds(
        self, t1: float, t2: float, pen: QtGui.QPen | None = None
    ) -> None:
        """Draw horizontal and vertical lines at 't1' and 't2'."""
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0, QtCore.Qt.DashedLine)
            pen.setCosmetic(True)

        line = pyqtgraph.PlotCurveItem([t1, 0.0], [t1, 1.0], pen=pen)
        self.plot.addItem(line)
        line = pyqtgraph.PlotCurveItem([0.0, t2], [1.0, t2], pen=pen)
        self.plot.addItem(line)
