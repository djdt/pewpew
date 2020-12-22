import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets


def float_image_to_indexed8(array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    data = np.clip(array, vmin, vmax)
    data = (data - vmin) / (vmax - vmin)
    data = data * 255.0
    return data.astype(np.uint8)


class NumpyImage(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        array: np.ndarray,
        extent: QtCore.QRectF = None,
        vmin: float = None,
        vmax: float = None,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(parent)

        if array.ndim != 2:
            raise ValueError("Array must be 2 dimensional.")

        if vmin is None:
            vmin = np.amin(array)
        if vmax is None:
            vmax = np.amax(array)
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        if extent is None:
            extent = QtCore.QRectF(0.0, 0.0, array.shape[1], array.shape[0])

        self.array = array
        self.vmin, self.vmax = vmin, vmax
        self.extent = extent
        self.data = float_image_to_indexed8(self.array, self.vmin, self.vmax)

        self.image = QtGui.QImage(
            self.data.data,
            self.data.shape[1],
            self.data.shape[0],
            self.data.strides[0],
            QtGui.QImage.Format_Indexed8,
        )

    def boundingRect(self) -> QtCore.QRectF:
        return self.extent

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):
        painter.drawImage(self.extent, self.image)
