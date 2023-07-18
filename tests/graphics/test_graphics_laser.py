import numpy as np
from pewlib.config import Config
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot
from testing import rand_data

from pewpew.graphics.imageitems import LaserImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.selectionitems import SelectionItem
from pewpew.graphics.widgetitems import ImageSliceWidgetItem, RulerWidgetItem


def left_click_at(pos: QtCore.QPointF) -> QtGui.QMouseEvent:
    return QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        pos,
        pos,
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )


def test_laser_graphics_selection(qtbot: QtBot):
    win = QtWidgets.QMainWindow()
    graphics = LaserGraphicsView(GraphicsOptions(), win)
    win.setCentralWidget(graphics)
    qtbot.addWidget(win)

    item = LaserImageItem(
        Laser(
            data=rand_data(["a", "b"]),
            info={"Name": "test"},
            config=Config(1.0, 1.0, 1.0),
        ),
        graphics.options,
    )
    graphics.scene().addItem(item)
    qtbot.waitExposed(graphics)

    # Test rectangle selector and center pixel selecting
    graphics.startRectangleSelection()

    event = left_click_at(graphics.mapFromScene(QtCore.QPointF(1, 1)))
    graphics.mousePressEvent(event)
    event = left_click_at(graphics.mapFromScene(QtCore.QPointF(10, 10)))
    graphics.mouseMoveEvent(event)
    graphics.mouseReleaseEvent(event)

    assert np.all(item.mask[1:, 1:])
    assert np.all(item.mask[1:, 0] == 0)
    assert np.all(item.mask[0, 1:] == 0)

    graphics.endSelection()
    assert not any(isinstance(item, SelectionItem) for item in graphics.scene().items())
    assert np.all(item.mask)

    # Test lasso works
    graphics.startLassoSelection()

    event = left_click_at(graphics.mapFromScene(QtCore.QPointF(1, 1)))
    graphics.mousePressEvent(event)
    event = left_click_at(graphics.mapFromScene(QtCore.QPointF(99, 1)))
    graphics.mouseMoveEvent(event)
    event = left_click_at(graphics.mapFromScene(QtCore.QPointF(99, 99)))
    graphics.mouseMoveEvent(event)
    event = left_click_at(graphics.mapFromScene(QtCore.QPointF(91, 99)))
    graphics.mouseMoveEvent(event)
    event = left_click_at(graphics.mapFromScene(QtCore.QPointF(91, 11)))
    graphics.mouseMoveEvent(event)
    event = left_click_at(graphics.mapFromScene(QtCore.QPointF(1, 11)))
    graphics.mouseMoveEvent(event)
    graphics.mouseReleaseEvent(event)

    assert np.all(item.mask[1:, 1:])
    assert np.all(item.mask[1:, 0] == 0)
    assert np.all(item.mask[0, 1:] == 0)


def test_laser_graphics_widgets(qtbot: QtBot):
    win = QtWidgets.QMainWindow()
    graphics = LaserGraphicsView(GraphicsOptions(), win)
    win.setCentralWidget(graphics)
    qtbot.addWidget(win)

    item = LaserImageItem(
        Laser(
            data=rand_data(["a", "b"]),
            info={"Name": "test"},
            config=Config(1.0, 1.0, 1.0),
        ),
        graphics.options,
    )
    graphics.scene().addItem(item)
    qtbot.waitExposed(graphics)


    graphics.startRulerWidget()
    event = left_click_at(graphics.mapFromScene(QtCore.QPointF(0, 0)))
    graphics.mousePressEvent(event)
    event = left_click_at(graphics.mapFromScene(QtCore.QPointF(100, 100)))
    graphics.mouseMoveEvent(event)
    graphics.mouseReleaseEvent(event)

    ruler = next(
        item for item in graphics.scene().items() if isinstance(item, RulerWidgetItem)
    )

    assert np.isclose(ruler.line.length(), 100 * np.sqrt(2))

    graphics.endWidget()
    assert not any(
        isinstance(item, RulerWidgetItem) for item in graphics.scene().items()
    )

    # This test crashes pytest

    # graphics.startSliceWidget()
    # slice = next(
    #     item
    #     for item in graphics.scene().items()
    #     if isinstance(item, ImageSliceWidgetItem)
    # )

    # event = left_click_at(graphics.mapFromScene(QtCore.QPointF(1, 1)))
    # graphics.mousePressEvent(event)
    # event = left_click_at(graphics.mapFromScene(QtCore.QPointF(95, 5)))
    # graphics.mouseMoveEvent(event)
    # graphics.mouseReleaseEvent(event)

    # print(slice.sliced)
    # assert np.all(slice.sliced == item.laser.data["a"][:, 0])


# def test_laser_graphics_zoom(qtbot: QtBot):
#     graphics = LaserGraphicsView(GraphicsOptions())
#     qtbot.addWidget(graphics)

#     x = np.random.random((10, 10))
#     graphics.drawImage(x, QtCore.QRectF(0, 0, 100, 100), "x")

#     qtbot.waitExposed(graphics)

#     graphics.zoomStart()

#     event = left_click_at(graphics.mapFromScene(QtCore.QPointF(20, 20)))
#     graphics.mousePressEvent(event)
#     event = left_click_at(graphics.mapFromScene(QtCore.QPointF(40, 40)))
#     graphics.mouseMoveEvent(event)
#     graphics.mouseReleaseEvent(event)

#     rect = graphics.mapToScene(graphics.viewport().rect()).boundingRect()
#     assert 29.5 < rect.center().x() < 30.5
#     assert 29.5 < rect.center().y() < 30.5
#     assert 19.5 < rect.width() < 20.5 or 19.5 < rect.height() < 20.5
