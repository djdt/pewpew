import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.charts.base import SinglePlotGraphicsView, ViewBoxForceScaleAtZero


class SpectraItem(pyqtgraph.PlotCurveItem):
    def __init__(self, xs, ys, *args, **kargs):

        xs = np.repeat(xs, 2)
        ys = np.stack((np.zeros_like(ys), ys), axis=1).ravel()

        super().__init__(xs, ys, *args, **kargs)
        self.text = pyqtgraph.TextItem(anchor=(0.5, 1.0))
        self.text.setParentItem(self)
        # self.text.setVisible(False)

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        if self.isUnderMouse():
            idx = np.argmin(np.abs(self.xData - event.pos().x()))
            x, y = self.xData[idx], self.yData[idx]
            if y == 0:
                y = self.yData[idx + 1]
            self.text.setPos(x, y)
            print(self.text.pos())
            self.text.setPlainText(f"{self.xData[idx]:.4g}")
            self.text.setVisible(True)
        else:
            self.text.setVisible(False)


class SpectraView(SinglePlotGraphicsView):
    mzClicked = QtCore.Signal(float)

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
        self.plot.setMouseEnabled(y=False)
        self.plot.setAutoVisible(y=True)
        self.plot.enableAutoRange(y=True)
        # self.plot.setAcceptHoverEvents(True)

        self.spectra: SpectraItem | None = None

    # def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
    #     self.text.setVisible(False)
    #     if self.spectra is not None:
    #         x, y = self.spectra.getData()
    #         idx = np.argmin(np.abs(x - event.pos().x()))
    #         if np.abs(x[idx] - event.pos().x()) < 0.3:
    #             self.text.setPos(x[idx], y[idx])
    #             self.text.setText(f"{x:.4f}")
    #
    #     super().hoverMoveEvent(event)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(640, 240)

    def drawCentroidSpectra(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)

        self.spectra = SpectraItem(
            x, y, pen=pen, connect="pairs", skipFiniteCheck=True
        )
        self.spectra.setAcceptHoverEvents(True)
        self.plot.addItem(self.spectra)
        #
        # brush = QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush)
        # pen = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
        #
        # hover_brush = QtGui.QBrush(QtCore.Qt.GlobalColor.red)
        #
        # scatter = pyqtgraph.ScatterPlotItem(
        #     x=x, y=y, pen=pen, brush=brush, hoverable=True, hoverBrush=hover_brush
        # )
        # # scatter.setAcceptHoverEvents(True)
        # # scatter.sigHovered.connect(lambda x: print(x))
        # self.plot.addItem(scatter)
        # print(scatter.acceptHoverEvents())
