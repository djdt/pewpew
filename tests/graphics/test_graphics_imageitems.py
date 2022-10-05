import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from pytestqt.qtbot import QtBot

from pewpew.graphics.imageitems import (
    ScaledImageItem,
    RulerWidgetItem,
    ImageSliceWidgetItem,
)


class FakeSceneMouseEvent(QtWidgets.QGraphicsSceneMouseEvent):
    def __init__(self, button: QtCore.Qt.MouseButton, pos: QtCore.QPointF):
        super().__init__()
        self._button = button
        self._pos = pos

    def button(self) -> QtCore.Qt.MouseButton:
        return self._button

    def buttons(self) -> QtCore.Qt.MouseButtons:
        return self._button

    def pos(self) -> QtCore.QPointF:
        return self._pos

    def scenePos(self) -> QtCore.QPointF:
        return self._pos


def test_scaled_image_item():
    image = QtGui.QImage(100, 100, QtGui.QImage.Format_Mono)
    image.fill(QtCore.Qt.white)

    item = ScaledImageItem(image, QtCore.QRectF(0, 100, 200, 400))

    assert item.width() == 100
    assert item.height() == 100
    assert item.pixelSize() == QtCore.QSizeF(2.0, 4.0)
    assert item.mapToData(QtCore.QPointF(100, 100)) == QtCore.QPoint(50, 0)


def test_scaled_image_item_smooth():
    image = QtGui.QImage(100, 100, QtGui.QImage.Format_Mono)
    image.fill(QtCore.Qt.white)

    item = ScaledImageItem(image, QtCore.QRectF(0, 100, 200, 400), smooth=True)

    assert item.width() == 100
    assert item.height() == 100
    assert item.pixelSize() == QtCore.QSizeF(2.0, 4.0)
    assert item.mapToData(QtCore.QPointF(100, 100)) == QtCore.QPoint(50, 0)


def test_scaled_image_item_from_array():
    array = np.random.random((10, 10))
    item = ScaledImageItem.fromArray(
        array, QtCore.QRectF(0, 0, 20, 20), colortable=[0, 1, 2, 3]
    )

    assert item.image.format() == QtGui.QImage.Format_Indexed8
    assert item.image.colorCount() == 4

    assert item.width() == 10
    assert item.height() == 10
    assert item.pixelSize() == QtCore.QSizeF(2.0, 2.0)


def test_ruler_widget_item(qtbot: QtBot):
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)

    scene = QtWidgets.QGraphicsScene(0, 0, 100, 100)
    view = QtWidgets.QGraphicsView(scene)
    window.setCentralWidget(view)

    img = ScaledImageItem.fromArray(
        np.random.random((10, 10)), QtCore.QRectF(50, 50, 50, 50)
    )
    item = RulerWidgetItem(img)
    scene.addItem(item)
    item.mousePressEvent(
        FakeSceneMouseEvent(QtCore.Qt.LeftButton, QtCore.QPointF(50, 50))
    )
    item.mouseMoveEvent(
        FakeSceneMouseEvent(QtCore.Qt.LeftButton, QtCore.QPointF(100, 100))
    )
    item.mouseReleaseEvent(
        FakeSceneMouseEvent(QtCore.Qt.LeftButton, QtCore.QPointF(100, 100))
    )

    assert item.line.length() == np.sqrt(50 ** 2 + 50 ** 2)

    # Draw everything
    window.show()
    qtbot.waitExposed(window)


def test_image_slice_item(qtbot: QtBot):
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)

    scene = QtWidgets.QGraphicsScene(0, 0, 100, 100)
    view = QtWidgets.QGraphicsView(scene)
    window.setCentralWidget(view)

    array = np.random.random((10, 10))
    img = ScaledImageItem.fromArray(array, QtCore.QRectF(0, 0, 100, 100))
    item = ImageSliceWidgetItem(img, array)
    scene.addItem(item)

    item.mousePressEvent(
        FakeSceneMouseEvent(QtCore.Qt.LeftButton, QtCore.QPointF(1, 1))
    )
    item.mouseMoveEvent(
        FakeSceneMouseEvent(QtCore.Qt.LeftButton, QtCore.QPointF(1, 99))
    )
    item.mouseReleaseEvent(
        FakeSceneMouseEvent(QtCore.Qt.LeftButton, QtCore.QPointF(1, 99))
    )

    assert item.sliced.size == 10
    assert np.all(item.sliced == array[:, 0])

    item.actionCopyToClipboard()
    item.shape()  # For redraw

    # Draw everything
    window.show()
    qtbot.waitExposed(window)
