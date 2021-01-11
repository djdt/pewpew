import numpy as np
from pytestqt.qtbot import QtBot
from PySide2 import QtCore, QtGui

from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.lasergraphicsview import LaserGraphicsView


def test_laser_graphics_selection(qtbot: QtBot):
    graphics = LaserGraphicsView(GraphicsOptions())
    qtbot.addWidget(graphics)

    x = np.random.random((10, 10))
    graphics.drawImage(x, QtCore.QRectF(0, 0, 100, 100), "x")

    qtbot.waitForWindowShown(graphics)

    # Test rectangle selector and center pixel selecting
    graphics.startRectangleSelection()

    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        graphics.mapFromScene(QtCore.QPointF(1, 1)),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )

    graphics.mousePressEvent(event)
    event.setLocalPos(graphics.mapFromScene(QtCore.QPointF(9, 9)))
    graphics.mouseMoveEvent(event)
    graphics.mouseReleaseEvent(event)

    assert graphics.mask[0][0]
    assert np.all(graphics.mask[1:, :] == 0)
    assert np.all(graphics.mask[:, 1:] == 0)

    graphics.endSelection()
    assert graphics.selection_item is None
    assert np.all(graphics.mask == 0)
    assert not graphics.posInSelection(graphics.mapFromScene(QtCore.QPoint(1, 1)))

    # Test lasso works
    graphics.startLassoSelection()

    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        graphics.mapFromScene(QtCore.QPointF(1, 1)),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )

    graphics.mousePressEvent(event)
    event.setLocalPos(graphics.mapFromScene(QtCore.QPointF(99, 1)))
    graphics.mouseMoveEvent(event)
    event.setLocalPos(graphics.mapFromScene(QtCore.QPointF(99, 99)))
    graphics.mouseMoveEvent(event)
    event.setLocalPos(graphics.mapFromScene(QtCore.QPointF(91, 99)))
    graphics.mouseMoveEvent(event)
    event.setLocalPos(graphics.mapFromScene(QtCore.QPointF(91, 11)))
    graphics.mouseMoveEvent(event)
    event.setLocalPos(graphics.mapFromScene(QtCore.QPointF(1, 11)))
    graphics.mouseMoveEvent(event)
    graphics.mouseReleaseEvent(event)

    assert np.all(graphics.mask[0, :])
    assert np.all(graphics.mask[:, -1])
    assert np.all(graphics.mask[1:, :-1] == 0)

    assert graphics.posInSelection(graphics.mapFromScene(QtCore.QPoint(1, 1)))
    assert not graphics.posInSelection(graphics.mapFromScene(QtCore.QPoint(11, 11)))


def test_laser_graphics_widgets(qtbot: QtBot):
    graphics = LaserGraphicsView(GraphicsOptions())
    qtbot.addWidget(graphics)

    x = np.random.random((10, 10))
    graphics.drawImage(x, QtCore.QRectF(0, 0, 100, 100), "x")

    qtbot.waitForWindowShown(graphics)

    graphics.startRulerWidget()
    assert graphics.widget is not None
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        graphics.mapFromScene(QtCore.QPointF(0, 0)),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    graphics.mousePressEvent(event)
    event.setLocalPos(graphics.mapFromScene(QtCore.QPointF(100, 100)))
    graphics.mouseMoveEvent(event)
    graphics.mouseReleaseEvent(event)

    assert np.isclose(graphics.widget.line.length(), 100 * np.sqrt(2))

    graphics.endWidget()
    assert graphics.widget is None

    graphics.startSliceWidget()
    assert graphics.widget is not None
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        graphics.mapFromScene(QtCore.QPointF(5, 5)),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    graphics.mousePressEvent(event)
    event.setLocalPos(graphics.mapFromScene(QtCore.QPointF(95, 5)))
    graphics.mouseMoveEvent(event)
    graphics.mouseReleaseEvent(event)

    assert np.all(graphics.widget.sliced == x[0, :])
