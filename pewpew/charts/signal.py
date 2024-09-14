import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.charts.base import SinglePlotGraphicsView


class SignalView(SinglePlotGraphicsView):
    """For drawing raw LAICPMS signals."""

    def __init__(
        self, title: str | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(
            "Signal", xlabel="Time", xunits="s", ylabel="Intensity", parent=parent
        )

        self.plot.setMouseEnabled(y=False)
        self.plot.setAutoVisible(y=True)
        self.plot.enableAutoRange(y=True)
        self.plot.setLimits(yMin=0.0)

        # self.series = {}

    def addLine(
        self,
        name: str,
        ys: np.ndarray,
        xs: np.ndarray | None = None,
        pen: QtGui.QPen | None = None,
    ) -> None:
        """Add a line plot to the chart.

        Args:
            name: key for series
            ys: y data
            xs: x data, defaults is range(ys.size)
            color: color of lines
            linewidth: width of lines
        """
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)

        if xs is None:
            xs = np.arange(ys.size)

        curve = pyqtgraph.PlotCurveItem(
            x=xs, y=ys, name=name, pen=pen, connect="all", skipFiniteCheck=True
        )
        self.plot.addItem(curve)

    def addScatterSeries(
        self,
        name: str,
        ys: np.ndarray,
        xs: np.ndarray,
        brush: QtGui.QBrush | None = None,
        color: QtGui.QColor = QtCore.Qt.black,
        markersize: float = 10.0,
    ) -> None:
        """Add a scatter plot to the chart.

        Args:
            name: key for series
            ys: y data
            xs: x data, defaults is range(ys.size)
            color: color of markers
            markersize: size of markers
        """
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.red)

        scatter = pyqtgraph.ScatterPlotItem(
            x=xs, y=ys, size=markersize, pen=None, brush=brush
        )
        self.plot.addItem(scatter)
