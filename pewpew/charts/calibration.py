import numpy as np
import pyqtgraph
from pewlib.calibration import weighted_linreg, weights_from_weighting
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
        self.line = None
        self.text = None
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.enableAutoRange(x=True, y=True)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(480, 480)

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
        pen: QtGui.QPen | None = None,
        brush: QtGui.QPen | None = None,
    ) -> None:
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.red)
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        hover_brush = QtGui.QBrush(brush)
        hover_brush.setColor(brush.color().lighter())

        if self.points is not None:
            self.plot.removeItem(self.points)

        self.points = pyqtgraph.ScatterPlotItem(
            points[:, 0],
            points[:, 1],
            symbol="o",
            size=10,
            pen=pen,
            brush=brush,
            hoverBrush=hover_brush,
            hoverable=True,
            tip="x: {x:.3g}\ny: {y:.3g}".format,
        )
        self.plot.addItem(self.points)

        if name is not None:
            self.plot.legend.addItem(self.points, name)

    def drawTrendline(self, weights: np.ndarray, pen: QtGui.QPen | None = None) -> None:
        if self.points is None:
            return

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.red, 1.0)
            pen.setCosmetic(True)

        x, y = self.points.getData()

        if x.size < 2 or np.all(x == x[0]):  # pragma: no cover
            return

        m, b, r2, err = weighted_linreg(x, y, w=weights)
        x0, x1 = x.min(), x.max()

        if self.line is None:
            self.line = pyqtgraph.PlotCurveItem(
                [x0, x1], [m * x0 + b, m * x1 + b], pen=pen
            )
            self.plot.addItem(self.line)
        else:
            self.line.setData([x0, x1], [m * x0 + b, m * x1 + b])

        if self.text is None:
            self.text = pyqtgraph.LabelItem(f"r² = {r2:.4f}", parent=self.yaxis)
            self.text.anchor(itemPos=(0, 0), parentPos=(1, 0), offset=(10, 10))
            self.text.setPos(x1, m * x1 + b)
        else:
            self.text.setText(f"r² = {r2:.4f}")
