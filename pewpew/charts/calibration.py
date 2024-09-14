import numpy as np
import pyqtgraph
from pewlib.calibration import weighted_linreg
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.charts.base import SinglePlotGraphicsView


class CalibrationView(SinglePlotGraphicsView):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Calibration", "Concentration", "Response", parent=parent)
        self.setMinimumSize(320, 320)

        self.points = None
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.enableAutoRange(x=True, y=True)

    def dataForExport(self) -> dict[str, np.ndarray]:
        if self.points is None:
            raise ValueError("no data for export")

        x, y = self.points.getData()
        return {"concentration": x, "response": y}

    def readyForExport(self) -> bool:
        return self.points is not None

    def drawPoints(
        self,
        points: np.ndarray,
        name: str | None = None,
        draw_trendline: bool = False,
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

        self.points = pyqtgraph.ScatterPlotItem(
            points[:, 0], points[:, 1], symbol="o", size=10, pen=pen, brush=brush
        )
        self.plot.addItem(self.points)

        if name is not None:
            self.plot.legend.addItem(self.points, name)

        if draw_trendline:
            pen = QtGui.QPen(brush.color(), 1.0)
            pen.setCosmetic(True)
            self.drawTrendline(pen=pen)

    def drawTrendline(
        self, weighting: str = "none", pen: QtGui.QPen | None = None
    ) -> None:
        if self.points is None:
            return

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.red, 1.0)
            pen.setCosmetic(True)

        x, y = self.points.getData()

        if weighting != "none":
            raise NotImplementedError("Weighting not yet implemented.")
        if x.size < 2 or np.all(x == x[0]):
            return

        m, b, r2, err = weighted_linreg(x, y, w=None)
        x0, x1 = x.min(), x.max()

        line = pyqtgraph.PlotCurveItem([x0, x1], [m * x0 + b, m * x1 + b], pen=pen)
        self.plot.addItem(line)

        text = pyqtgraph.LabelItem(f"r² = {r2:.4f}", parent=self.yaxis)
        text.anchor(itemPos=(0, 0), parentPos=(1, 0), offset=(10, 10))
        text.setPos(x1, m * x1 + b)
