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

from testing import FakeEvent, FakePick


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
    canvas.ax = canvas.figure.subplots()
    qtbot.addWidget(canvas)
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    event = FakeEvent(canvas.ax, 0.0, 0.0)
    pick_event = FakePick(canvas.ax, 0.0, 0.0)

    assert not canvas.ignore_event(event)
    with pytest.raises(NotImplementedError):
        canvas._axis_enter(event)
    with pytest.raises(NotImplementedError):
        canvas._axis_leave(event)
    with pytest.raises(NotImplementedError):
        canvas._press(event)
    with pytest.raises(NotImplementedError):
        canvas._release(event)
    with pytest.raises(NotImplementedError):
        canvas._keypress(event)
    with pytest.raises(NotImplementedError):
        canvas._move(event)
    with pytest.raises(NotImplementedError):
        canvas._pick(pick_event)
    with pytest.raises(NotImplementedError):
        canvas._scroll(event)

    class FakeWidget:
        def get_active(self) -> bool:
            return True

    canvas.widget = FakeWidget()
    assert canvas.ignore_event(event)
    canvas._axis_enter(event)
    canvas._axis_leave(event)
    canvas._press(event)
    canvas._release(event)
    canvas._keypress(event)
    canvas._move(event)
    canvas._pick(pick_event)
    canvas._scroll(event)

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

    # Check no crash with no window / status bar missing
    canvas._move(FakeEvent(canvas.ax, 30.0, 30.0))


def test_canvas_interactive_laser_events(qtbot: QtBot):
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)
    canvas = InteractiveLaserCanvas(ViewOptions())
    window.setCentralWidget(canvas)
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    laser = Laser(np.array(np.random.random((10, 10)), dtype=[("a", float)]))
    canvas.drawLaser(laser, "a")
    canvas.draw()

    # Point under cursor
    assert window.statusBar().currentMessage() == ""
    canvas._move(FakeEvent(canvas.ax, 30.0, 30.0))
    assert window.statusBar().currentMessage() == f"30,30 [{laser.data['a'][9, 0]:.4g}]"
    canvas.viewoptions.units = "row"
    canvas._move(FakeEvent(canvas.ax, 40.0, 40.0))
    assert window.statusBar().currentMessage() == f"8,1 [{laser.data['a'][8, 1]:.4g}]"
    canvas.viewoptions.units = "second"
    canvas._move(FakeEvent(canvas.ax, 30.0, 30.0))
    assert (
        window.statusBar().currentMessage() == f"0.2143,0 [{laser.data['a'][9, 0]:.4g}]"
    )
    canvas.axis_leave(None)
    assert window.statusBar().currentMessage() == ""

    # Scroll and drag
    canvas._scroll(FakeEvent(canvas.ax, 0.0, 0.0, step=1))
    assert canvas.view_limits == (0.0, 315.0, 0.0, 315.0)
    # Towards edge
    canvas._press(FakeEvent(canvas.ax, 0.0, 0.0))
    canvas._move(FakeEvent(canvas.ax, 10.0, 10.0))
    assert canvas.view_limits == (0.0, 315.0, 0.0, 315.0)
    # Away from edge
    canvas._press(FakeEvent(canvas.ax, 20.0, 20.0))
    canvas._move(FakeEvent(canvas.ax, 10.0, 10.0))
    assert canvas.view_limits == (10.0, 325.0, 10.0, 325.0)
    canvas.unzoom()
    assert canvas.view_limits == (0.0, 350.0, 0.0, 350.0)

    # Zoom
    canvas.startZoom()
    canvas.widget.press(FakeEvent(canvas.widget.ax, 100.0, 100.0))
    canvas.widget.onmove(FakeEvent(canvas.widget.ax, 300.0, 300.0))
    canvas.widget.release(FakeEvent(canvas.widget.ax, 300.0, 300.0))
    assert canvas.view_limits == (100.0, 300.0, 100.0, 300.0)
    canvas.unzoom()
    assert canvas.view_limits == (0.0, 350.0, 0.0, 350.0)


def test_canvas_interactive_laser_selections(qtbot: QtBot):
    canvas = InteractiveLaserCanvas(ViewOptions())
    qtbot.addWidget(canvas)
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    np.random.seed(99586566)
    laser = Laser(np.array(np.random.random((10, 10)), dtype=[("a", float)]))
    canvas.drawLaser(laser, "a")
    canvas.draw()

    # Ruler
    canvas.startRuler()
    canvas.endSelection()  # Clears widget

    # Selections
    assert canvas.selection_image is None
    # Test lassoo and ignored events
    canvas.startLassoSelection()
    assert canvas.ignore_event(FakeEvent(canvas.ax, 0.0, 0.0))
    canvas.clearSelection()  # Clears widget

    # Test rect and selections
    mask = np.zeros((10, 10), dtype=bool)
    mask[9, 0] = True

    canvas.startRectangleSelection()
    canvas.widget.press(FakeEvent(canvas.ax, 10.0, 10.0))
    canvas.widget.onmove(FakeEvent(canvas.ax, 40.0, 40.0))
    canvas.widget.release(FakeEvent(canvas.ax, 40.0, 40.0))

    assert canvas.selection_image is not None
    assert np.all(canvas.getSelection() == mask)
    assert np.all(canvas.getMaskedData() == laser.data["a"][9, 0])

    canvas.endSelection()  # Clears widget
    assert np.all(canvas.getSelection() == mask)
    canvas.clearSelection()
    assert canvas.selection is None

    # Test masked data for multiple selections
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    mask[2, 2] = True
    canvas.setSelection(mask)
    assert np.all(canvas.getSelection() == mask)
    masked_data = canvas.getMaskedData()
    assert masked_data.shape == (3, 3)

    assert np.all(np.isnan(masked_data) == ~mask[0:3, 0:3])

    class FakeQtKeyEscape(object):
        def key(self):
            return QtCore.Qt.Key_Escape

        def modifiers(self):
            return QtCore.Qt.NoModifier

    canvas.keyPressEvent(FakeQtKeyEscape())
    assert canvas.selection is None
