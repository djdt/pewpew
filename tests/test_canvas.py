import numpy as np
from pathlib import Path
import tempfile

from pytestqt.qtbot import QtBot

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib import Laser

from pewpew.lib.viewoptions import ViewOptions
from pewpew.widgets.canvases import (
    BasicCanvas,
    ImageCanvas,
    InteractiveImageCanvas,
    SelectableImageCanvas,
    LaserImageCanvas,
)

from matplotlib.pyplot import imread
from testing import FakeEvent


def test_canvas_basic(qtbot: QtBot):
    canvas = BasicCanvas()
    qtbot.addWidget(canvas)
    canvas.show()

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
    path = Path(__file__).parent.joinpath("data", "basic_canvas_clipboard.png")
    actual = QtWidgets.QApplication.clipboard().pixmap().toImage()
    expected = QtGui.QImage(str(path.resolve())).convertToFormat(actual.format())
    assert actual == expected

    # Test context menu
    canvas.contextMenuEvent(
        QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0))
    )
    canvas.close()


def test_image_canvas(qtbot: QtBot):
    canvas = ImageCanvas(figsize=(1.0, 1.0))
    qtbot.addWidget(canvas)
    canvas.drawFigure()
    canvas.show()

    np.random.seed(89328634)
    canvas.image = canvas.ax.imshow(
        np.random.random((10, 10)),
        extent=(0, 100, 0, 200),
        aspect=1,
        origin="upper",
        interpolation="none",
    )
    canvas.view_limits = canvas.extentForAspect(canvas.extent)
    qtbot.waitForWindowShown(canvas)
    canvas.draw()

    # Test aspect fitting

    assert canvas.view_limits == (-50.0, 150.0, 0.0, 200.0)
    # No change
    canvas.resize(200, 200)
    assert canvas.view_limits == (-50.0, 150.0, 0.0, 200.0)
    # Reduce border
    canvas.resize(100, 200)
    assert canvas.view_limits == (0.0, 100.0, 0.0, 200.0)
    # Reduce border
    canvas.resize(200, 100)
    assert canvas.view_limits == (-150.0, 250.0, 0.0, 200.0)
    # Change when zoomed
    canvas.view_limits = (10, 30, 10, 20)
    canvas.resize(100, 100)
    assert canvas.view_limits == (15, 25, 10, 20)
    canvas.resize(100, 200)
    assert canvas.view_limits == (15, 25, 5, 25)

    # Restore view lims on redraw
    canvas.drawFigure()
    assert canvas.view_limits == (15, 25, 5, 25)

    # Test image is correct
    path = Path(__file__).parent.joinpath("data", "image_canvas_raw.png").resolve()
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        canvas.saveRawImage(tf.name)
        assert np.all(imread(tf.name) == imread(str(path)))


def test_canvas_interactive_image(qtbot: QtBot):
    canvas = InteractiveImageCanvas(move_button=1)
    canvas.drawFigure()
    qtbot.addWidget(canvas)
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    canvas.image = canvas.ax.imshow(
        np.random.random((10, 10)),
        extent=(0, 100, 0, 100),
        aspect=1,
        origin="upper",
        interpolation="none",
    )
    canvas.view_limits = (0.0, 100.0, 0.0, 100.0)

    # Scroll and drag
    canvas._scroll(FakeEvent(canvas.ax, 0.0, 0.0, step=1))
    assert canvas.view_limits == (0.0, 90.0, 0.0, 90.0)
    # Towards edge
    canvas._press(FakeEvent(canvas.ax, 0.0, 0.0))
    canvas._move(FakeEvent(canvas.ax, 5.0, 5.0))
    assert canvas.view_limits == (0.0, 90.0, 0.0, 90.0)
    # Away from edge
    canvas._press(FakeEvent(canvas.ax, 10.0, 10.0))
    canvas._move(FakeEvent(canvas.ax, 5.0, 5.0))
    assert canvas.view_limits == (5.0, 95.0, 5.0, 95.0)

    # Zoom
    canvas.zoom(FakeEvent(canvas.ax, 10.0, 10.0), FakeEvent(canvas.ax, 20.0, 40.0))
    assert canvas.view_limits == (0.0, 30.0, 10.0, 40.0)  # Fixed aspect
    canvas.unzoom()
    assert canvas.view_limits == (0.0, 100.0, 0.0, 100.0)

    # Cursor status
    class FakeStatus(object):
        def setValue(self, x: float, y: float, value: float) -> None:
            self.value = value

        def clear(self) -> None:
            self.value = None

    status = FakeStatus()
    canvas.cursorMoved.connect(status.setValue)
    canvas.cursorClear.connect(status.clear)

    canvas._move(FakeEvent(canvas.ax, 5.0, 5.0))
    assert status.value == canvas.image.get_array()[-1, 0]
    canvas._axes_leave(FakeEvent(canvas.ax, 0.0, 0.0))
    assert status.value is None


def test_canvas_selectable_image(qtbot: QtBot):
    canvas = SelectableImageCanvas()
    canvas.drawFigure()
    qtbot.addWidget(canvas)
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    canvas.image = canvas.ax.imshow(
        np.random.random((10, 10)),
        extent=(0, 100, 0, 100),
        aspect=1,
        origin="upper",
        interpolation="none",
    )
    canvas.view_limits = (0.0, 100.0, 0.0, 100.0)

    canvas.selection = np.zeros((10, 10), dtype=bool)
    canvas.selection[2:8, 2:8] = True
    canvas.drawSelection()

    assert np.all(canvas.getMaskedData() == canvas.image.get_array()[2:8, 2:8])
    canvas.clearSelection()
    assert canvas.getMaskedData().shape == (10, 10)


def test_canvas_laser_image(qtbot: QtBot):
    canvas = LaserImageCanvas(ViewOptions())
    qtbot.addWidget(canvas)
    canvas.drawFigure()
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    np.random.seed(99586566)
    laser = Laser(np.array(np.random.random((10, 10)), dtype=[("a", float)]))
    canvas.drawLaser(laser, "a")

    assert canvas.extent == laser.extent

    canvas.viewoptions.canvas.label = False
    canvas.viewoptions.canvas.scalebar = False
    canvas.viewoptions.canvas.colorbar = False

    canvas.drawLaser(laser, "a")
