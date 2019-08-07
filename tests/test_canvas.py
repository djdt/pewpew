import os.path
import numpy as np

from pytestqt.qtbot import QtBot

from PySide2 import QtGui, QtWidgets

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
    canvas.close()
    # Test that the image generated is the same
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "basic_canvas_clipboard.png"
    )
    actual = QtWidgets.QApplication.clipboard().pixmap().toImage()
    expected = QtGui.QImage(data_path).convertToFormat(actual.format())
    assert actual == expected


def test_canvas_interactive(qtbot: QtBot):
    canvas = InteractiveCanvas()
    qtbot.addWidget(canvas)
    canvas.show()
    # Can only really check cids
    assert len(canvas.cids) == 7
    canvas.disconnect_events()
    assert len(canvas.cids) == 0
