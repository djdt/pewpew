import os.path
import numpy as np
import filecmp
import tempfile

import pytest
from pytestqt.qtbot import QtBot

from PySide2 import QtCore, QtGui, QtWidgets

from pew import Laser

from pewpew.lib.viewoptions import ViewOptions
from pewpew.widgets.canvases import (
    BasicCanvas,
    InteractiveCanvas,
    LaserCanvas,
    InteractiveLaserCanvas,
)

from testing import FakeEvent


def test_canvas_basic(qtbot: QtBot):
    canvas = BasicCanvas()
    qtbot.addWidget(canvas)
    canvas.show()
    canvas.redrawFigure()

    ax = canvas.figure.subplots()
    np.random.seed(11636971)
    ax.imshow(
        np.random.random((5, 5)),
        cmap="gray",
        interpolation="none",
        origin="upper",
        aspect="equal",
    )
    qtbot.waitForWindowShown(canvas)
    canvas.draw()
    canvas.copyToClipboard()
    # Test that the image generated is the same
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "basic_canvas_clipboard.png"
    )
    actual = QtWidgets.QApplication.clipboard().pixmap().toImage()
    expected = QtGui.QImage(data_path).convertToFormat(actual.format())
    assert actual == expected

    # Test context menu
    canvas.contextMenuEvent(
        QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0))
    )
    canvas.close()


def test_canvas_interactive(qtbot: QtBot):
    canvas = InteractiveCanvas()
    qtbot.addWidget(canvas)
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    class FakePick(object):
        mouseevent = None
        artist = None

    assert not canvas.ignore_event(None)
    with pytest.raises(NotImplementedError):
        canvas._axis_enter(None)
    with pytest.raises(NotImplementedError):
        canvas._axis_leave(None)
    with pytest.raises(NotImplementedError):
        canvas._press(None)
    with pytest.raises(NotImplementedError):
        canvas._release(None)
    with pytest.raises(NotImplementedError):
        canvas._keypress(None)
    with pytest.raises(NotImplementedError):
        canvas._move(None)
    with pytest.raises(NotImplementedError):
        canvas._pick(FakePick())
    with pytest.raises(NotImplementedError):
        canvas._scroll(None)

    class FakeWidget:
        def get_active(self) -> bool:
            return True

    canvas.widget = FakeWidget()
    assert canvas.ignore_event(None)
    canvas._axis_enter(None)
    canvas._axis_leave(None)
    canvas._press(None)
    canvas._release(None)
    canvas._keypress(None)
    canvas._move(None)
    canvas._pick(FakePick())
    canvas._scroll(None)

    # Can only really check cids
    assert len(canvas.cids) == 8
    canvas.disconnect_events()
    assert len(canvas.cids) == 0


def test_canvas_laser(qtbot: QtBot):
    canvas = LaserCanvas(ViewOptions())
    qtbot.addWidget(canvas)
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    assert canvas.extent == (0, 0, 0, 0)

    np.random.seed(99586566)
    laser = Laser(np.array(np.random.random((10, 10)), dtype=[("a", float)]))
    canvas.drawLaser(laser, "a")

    assert canvas.extent == laser.extent

    # Test image is correct
    data_path = os.path.join(os.path.dirname(__file__), "data", "laser_canvas_raw.png")
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        canvas.saveRawImage(tf.name)
        assert filecmp.cmp(tf.name, data_path)

    canvas.viewoptions.canvas.label = False
    canvas.viewoptions.canvas.scalebar = False
    canvas.viewoptions.canvas.colorbar = False

    canvas.drawLaser(laser, "a")


def test_canvas_interactive_laser(qtbot: QtBot):
    canvas = InteractiveLaserCanvas(ViewOptions())
    qtbot.addWidget(canvas)
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    np.random.seed(99586566)
    laser = Laser(np.array(np.random.random((10, 10)), dtype=[("a", float)]))
    canvas.drawLaser(laser, "a")
    canvas.draw()

    # Test image is correct
    data_path = os.path.join(os.path.dirname(__file__), "data", "laser_canvas_raw.png")
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        canvas.saveRawImage(tf.name)
        assert filecmp.cmp(tf.name, data_path)

    # Point under cursor
    # Zoom
    assert canvas.view_limits == (0.0, 350.0, 0.0, 350.0)
    canvas.startZoom()
    canvas.widget.press(FakeEvent(canvas.widget.ax, 100.0, 100.0))
    canvas.widget.onmove(FakeEvent(canvas.widget.ax, 300.0, 300.0))
    canvas.widget.release(FakeEvent(canvas.widget.ax, 300.0, 300.0))
    assert canvas.view_limits == (100.0, 300.0, 100.0, 300.0)
    canvas.unzoom()
    assert canvas.view_limits == (0.0, 350.0, 0.0, 350.0)
