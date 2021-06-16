import numpy as np
from PySide2 import QtWidgets

from pytestqt.qtbot import QtBot

from pewpew.graphics.colortable import get_table
from pewpew.graphics.overlayitems import ColorBarOverlay


def test_color_bar_overlay(qtbot: QtBot):
    scene = QtWidgets.QGraphicsScene(0, 0, 100, 100)
    view = QtWidgets.QGraphicsView(scene)
    qtbot.addWidget(view)

    table = get_table("grey")
    cb = ColorBarOverlay(table, vmin=0.0, vmax=10.0)
    scene.addItem(cb)

    # Format values
    assert np.all(cb.niceTextValues(5) == [0, 2, 4, 6, 8])
    assert np.all(cb.niceTextValues(7) == [0, 1.5, 3, 4.5, 6, 7.5, 9])
    assert np.all(cb.niceTextValues(5, 1) == [1.5, 3, 4.5, 6, 7.5])
    cb.updateTable(table, 100.0, 10100.0)
    assert np.all(cb.niceTextValues(5) == [0, 2000, 4000, 6000, 8000])
