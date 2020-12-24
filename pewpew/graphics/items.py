from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np


class ScaledImageItem(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        image: QtGui.QImage,
        rect: QtCore.QRectF,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(parent)
        self.image = image
        self.rect = rect
        self._data = None

    def boundingRect(self) -> QtCore.QRectF:
        return self.rect

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):
        painter.drawImage(self.rect, self.image)

    @classmethod
    def fromArray(
        cls,
        array: np.ndarray,
        rect: QtCore.QRectF,
        image_format: QtGui.QImage.Format,
        colortable: np.ndarray = None,
        parent: QtWidgets.QGraphicsItem = None,
    ) -> "ScaledImageItem":
        image = QtGui.QImage(
            array.data, array.shape[1], array.shape[0], array.strides[0], image_format
        )
        if colortable is not None:
            image.setColorTable(colortable)
        item = cls(image, rect, parent=parent)
        item._data = array
        return item
