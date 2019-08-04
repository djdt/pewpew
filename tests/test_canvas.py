from pytestqt.qtbot import QtBot
from PySide2 import QtWidgets
import os.path
import tempfile
from hashlib import md5
import numpy as np

from pewpew.ui.canvas.basic import BasicCanvas
from pewpew.ui.canvas.interactive import InteractiveCanvas
from pewpew.ui.canvas.laser import LaserCanvas


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
    temp = tempfile.NamedTemporaryFile()
    QtWidgets.QApplication.clipboard().pixmap().save(temp.name, "png")
    temp.seek(0)
    with open(os.path.join(data_path), 'rb') as fp:
        assert md5(temp.read()).digest() == md5(fp.read()).digest()
    temp.close()


def test_canvas_interactive(qtbot: QtBot):
    canvas = InteractiveCanvas()
    qtbot.addWidget(canvas)
    canvas.show()
    # Can only really check cids
    assert len(canvas.cids) == 7
    canvas.disconnect_events()
    assert len(canvas.cids) == 0


# def test_laser_canvas(qtbot: QtBot):
#     canvas = LaserCanvas()
#     qtbot.addWidget(canvas)
#     canvas.show()
