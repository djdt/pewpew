import os.path
import numpy as np
import filecmp
import tempfile

from pytestqt.qtbot import QtBot

from PySide2 import QtGui, QtWidgets

from pew import Laser

from pewpew.lib.viewoptions import ViewOptions
from pewpew.widgets.canvases import BasicCanvas, InteractiveCanvas, LaserCanvas


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
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "basic_canvas_clipboard.png"
    )
    actual = QtWidgets.QApplication.clipboard().pixmap().toImage()
    expected = QtGui.QImage(data_path).convertToFormat(actual.format())
    assert actual == expected
    canvas.close()


def test_canvas_laser(qtbot: QtBot):
    canvas = LaserCanvas(ViewOptions())
    qtbot.addWidget(canvas)
    canvas.show()

    np.random.seed(99586566)
    laser = Laser(np.array(np.random.random((10, 10)), dtype=[("a", float)]))

    canvas.drawLaser(laser, "a")
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "laser_canvas_raw.png"
    )
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        canvas.saveRawImage(tf.name)
        assert filecmp.cmp(tf.name, data_path)


def test_canvas_interactive(qtbot: QtBot):
    canvas = InteractiveCanvas()
    qtbot.addWidget(canvas)
    canvas.show()
    # Can only really check cids
    assert len(canvas.cids) == 8
    canvas.disconnect_events()
    assert len(canvas.cids) == 0


def test_canvas_interactive_laser(qtbot: QtBot):
    pass
