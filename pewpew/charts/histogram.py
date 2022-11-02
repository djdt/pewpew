from PySide6 import QtCore, QtGui, QtWidgets
from PySide6 import QtCharts

import numpy as np

from pewpew.charts.base import BaseChart
from pewpew.charts.colors import light_theme, sequential


class HistogramChart(BaseChart):
    """BaseChart for drawing a histogram."""

    def __init__(
        self, title: str | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(QtCharts.QChart(), theme=light_theme, parent=parent)
        self.setMinimumSize(QtCore.QSize(640, 320))
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        if title is not None:
            self.chart().setTitle(title)

        self.chart().legend().hide()

        self._xaxis = QtCharts.QValueAxis()  # Alignment axis
        self._xaxis.setVisible(False)
        self.xaxis = QtCharts.QValueAxis()  # Value axis
        self.xaxis.setGridLineVisible(False)

        self.yaxis = QtCharts.QValueAxis()
        self.yaxis.setGridLineVisible(False)
        self.yaxis.setLabelFormat("%d")

        self.addAxis(self._xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.series = QtCharts.QBarSeries()
        self.series.setBarWidth(1.0)

        self.chart().addSeries(self.series)
        self.series.attachAxis(self._xaxis)
        self.series.attachAxis(self.yaxis)

    def setHistogram(
        self,
        data: np.ndarray,
        bins: int | str = "auto",
        min_bins: int = 16,
        max_bins: int = 128,
    ) -> None:
        """Draw 'data' as a histogram.

        Args:
            data: hist data
            bins: passed to np.histogram_bin_edges
            min_bins: minimum number of bins
            max_bins: maximum number of bins
        """
        vmin, vmax = np.percentile(data, 5), np.percentile(data, 95)

        barset = QtCharts.QBarSet("histogram")
        barset.setColor(sequential[1])
        barset.setLabelColor(light_theme["text"])

        bin_edges = np.histogram_bin_edges(data, bins=bins, range=(vmin, vmax))
        if bin_edges.size > max_bins:
            bin_edges = np.histogram_bin_edges(data, bins=max_bins, range=(vmin, vmax))
        elif bin_edges.size < min_bins:
            bin_edges = np.histogram_bin_edges(data, bins=min_bins, range=(vmin, vmax))

        hist, edges = np.histogram(data, bins=bin_edges)
        barset.append(list(hist))

        self.series.clear()
        self.series.append(barset)

        self._xaxis.setRange(-0.5, hist.size - 0.5)
        self.xaxis.setRange(edges[0], edges[-1])
        self.yaxis.setRange(0, np.amax(hist))
        self.yaxis.applyNiceNumbers()
