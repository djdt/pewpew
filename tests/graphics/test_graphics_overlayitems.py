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

    cb.updateTable(table, 100.0, 10100.0, "ppm")


def test_metric_scalebar_overlay(qtbot: QtBot):
    scene = QtWidgets.QGraphicsScene(0, 0, 100, 100)
    view = QtWidgets.QGraphicsView(scene)
    qtbot.addWidget(view)
    with qtbot.wait_exposed(view):
        view.show()

    sb = MetricScaleBarOverlay()
    scene.addItem(sb)

    width, unit = sb.getWidthAndUnit(1.0, sb.unit)
    assert np.isclose(width, 0.5)
    assert unit == "nm"

    width, unit = sb.getWidthAndUnit(2.0, sb.unit)
    assert np.isclose(width, 1.0)
    assert unit == "μm"

    width, unit = sb.getWidthAndUnit(4.0, sb.unit)
    assert np.isclose(width, 2.0)
    assert unit == "μm"

    width, unit = sb.getWidthAndUnit(100.0, sb.unit)
    assert np.isclose(width, 50.0)
    assert unit == "μm"

    width, unit = sb.getWidthAndUnit(1000.0, sb.unit)
    assert np.isclose(width, 500.0)
    assert unit == "μm"
