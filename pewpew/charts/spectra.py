import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.charts.base import SinglePlotGraphicsView, ViewBoxForceScaleAtZero
from pewpew.lib.numpyqt import array_to_polygonf, polygonf_to_array


class SpectraItem(pyqtgraph.PlotCurveItem):
    mzClicked = QtCore.Signal(float)
    mzDoubleClicked = QtCore.Signal(float)

    def __init__(self, xs, ys, *args, **kargs):

        xs = np.repeat(xs, 2)
        ys = np.stack((np.zeros_like(ys), ys), axis=1).ravel()

        super().__init__(xs, ys, *args, **kargs)
        self.setAcceptHoverEvents(True)
        self.setFiltersChildEvents(True)

        self.text = pyqtgraph.TextItem(anchor=(0.0, 0.5))
        self.text.setParentItem(self)
        self.text.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.text.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent
        )

    def closestMz(self, pos: QtCore.QPointF) -> int:
        pos = self.mapToDevice(pos)

        poly = array_to_polygonf(np.stack((self.xData[1::2], self.yData[1::2]), axis=1))
        poly = self.mapToDevice(poly)
        arr = polygonf_to_array(poly)
        dist = np.square(arr[:, 0] - pos.x()) + np.square(arr[:, 1] - pos.y())
        return int(np.argmin(dist) * 2)

    def mouseClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self.mouseShape().contains(event.pos()):
            idx = self.closestMz(event.pos())
            self.mzClicked.emit(self.xData[idx])
            event.accept()
        else:
            super().mouseClickEvent(event)

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self.mouseShape().contains(event.pos()):
            idx = self.closestMz(event.pos())
            self.mzDoubleClicked.emit(self.xData[idx])

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        if self.mouseShape().contains(event.pos()):
            idx = self.closestMz(event.pos())
            x, y = self.xData[idx], self.yData[idx]
            if y == 0:
                y = self.yData[idx + 1]
            self.text.setPos(x, y)
            self.text.setPlainText(f"{self.xData[idx]:.4g}")
            self.text.setVisible(True)
            event.accept()
        else:
            self.text.setVisible(False)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self.text.setVisible(False)

    # Make some room for the text
    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        bounds = super().dataBounds(ax, frac, orthoRange)
        if bounds[1] is not None:
            bounds = bounds[0], bounds[1] * 1.1
        return bounds


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

        self.spectra: SpectraItem | None = None

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(640, 240)

    def readyForExport(self) -> bool:
        if self.spectra is None:
            return False
        if self.spectra.xData.size == 0:
            return False
        return True

    def dataForExport(self) -> dict[str, np.ndarray]:
        assert self.spectra is not None
        return {"m/z": self.spectra.xData, "signal": self.spectra.yData}

    def drawCentroidSpectra(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> SpectraItem:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)

        self.spectra = SpectraItem(x, y, pen=pen, connect="pairs", skipFiniteCheck=True)
        self.setLimits(xMin=np.nanmin(x), xMax=np.nanmax(x))
        self.plot.addItem(self.spectra)
        # self.setDataLimits(0.0, 1.0)
        return self.spectra
