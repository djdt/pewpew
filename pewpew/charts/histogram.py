import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.charts.base import SinglePlotGraphicsView, ViewBoxForceScaleAtZero


class HistogramView(SinglePlotGraphicsView):
    """BaseChart for drawing a histogram."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            "Histogram",
            xlabel="Signal (counts)",
            ylabel="No. Events",
            viewbox=ViewBoxForceScaleAtZero(),
            parent=parent,
        )
        self.plot.setLimits(yMin=0.0)
        self.hist: np.ndarray | None = None
        self.edges: np.ndarray | None = None

    def dataForExport(self) -> dict[str, np.ndarray]:
        if self.hist is None or self.edges is None:
            raise ValueError("not ready for export")
        return {"counts": self.hist, "bin_edges": self.edges}

    def readyForExport(self) -> bool:
        return self.hist is not None and self.edges is not None

    def setHistogram(
        self,
        data: np.ndarray,
        bins: int | str = "auto",
        min_bins: int = 16,
        max_bins: int = 128,
        bar_width: float = 0.5,
        bar_offset: float = 0.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> None:
        """Draw 'data' as a histogram.

        Args:
            data: hist data
            bins: passed to np.histogram_bin_edges
            min_bins: minimum number of bins
            max_bins: maximum number of bins
        """
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.black)

        assert bar_width > 0.0 and bar_width <= 1.0
        assert bar_offset >= 0.0 and bar_offset < 1.0

        vmin, vmax = np.percentile(data, 5), np.percentile(data, 95)

        bin_edges = np.histogram_bin_edges(data, bins=bins, range=(vmin, vmax))
        if bin_edges.size > max_bins:
            bin_edges = np.histogram_bin_edges(data, bins=max_bins, range=(vmin, vmax))
        elif bin_edges.size < min_bins:
            bin_edges = np.histogram_bin_edges(data, bins=min_bins, range=(vmin, vmax))

        self.hist, self.edges = np.histogram(data, bins=bin_edges)

        widths = np.diff(self.edges)
        x = np.repeat(self.edges, 2)

        # Calculate bar start and end points for width / offset
        x[1:-1:2] += widths * ((1.0 - bar_width) / 2.0 + bar_offset)
        x[2::2] -= widths * ((1.0 - bar_width) / 2.0 - bar_offset)

        y = np.zeros(self.hist.size * 2 + 1, dtype=self.hist.dtype)
        y[1:-1:2] = self.hist

        curve = pyqtgraph.PlotCurveItem(
            x=x,
            y=y,
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            pen=pen,
            brush=brush,
            skipFiniteCheck=True,
        )

        self.plot.addItem(curve)
