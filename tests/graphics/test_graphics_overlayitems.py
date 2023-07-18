import numpy as np
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from pewpew.graphics.colortable import get_table
from pewpew.graphics.overlayitems import ColorBarOverlay, MetricScaleBarOverlay


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
    cb.updateTable(table, 100.0, 10100.0, "ppm")
    assert np.all(cb.niceTextValues(4) == [0, 2500, 5000, 7500, 10000])


def test_metric_scalebar_overlay(qtbot: QtBot):
    scene = QtWidgets.QGraphicsScene(0, 0, 100, 100)
    view = QtWidgets.QGraphicsView(scene)
    qtbot.addWidget(view)

    sb = MetricScaleBarOverlay()
    scene.addItem(sb)

    width, unit = sb.getWidthAndUnit(1.0)
    assert np.isclose(width, 0.5)
    assert unit == "nm"

    width, unit = sb.getWidthAndUnit(2.0)
    assert np.isclose(width, 1.0)
    assert unit == "μm"

    width, unit = sb.getWidthAndUnit(4.0)
    assert np.isclose(width, 2.0)
    assert unit == "μm"

    width, unit = sb.getWidthAndUnit(100.0)
    assert np.isclose(width, 50.0)
    assert unit == "μm"

    width, unit = sb.getWidthAndUnit(1000.0)
    assert np.isclose(width, 500.0)
    assert unit == "μm"

    view.scale(10.0, 10.0)

    width, unit = sb.getWidthAndUnit(1000.0)
    assert np.isclose(width, 500.0)
    assert unit == "μm"
