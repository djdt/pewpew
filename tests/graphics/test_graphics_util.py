import numpy as np
from PySide6 import QtCore, QtGui
from pytestqt.qtbot import QtBot

from pewpew.graphics.util import (
    closest_nice_value,
    nice_values,
    path_for_colorbar_labels,
    polygonf_contains_points,
    shortest_label,
)


def test_polygonf_contains_points():
    X, Y = np.mgrid[:3, :3]
    a = np.stack((X.flat, Y.flat), axis=1)
    a = a.astype(np.float64)

    p = QtGui.QPolygonF()
    p.append(QtCore.QPointF(0.5, 0.5))
    p.append(QtCore.QPointF(3, 3))
    p.append(QtCore.QPointF(0.5, 3))

    r = polygonf_contains_points(p, a)
    assert np.all(r == [0, 0, 0, 0, 1, 1, 0, 0, 1])


def test_closest_nice_value():
    assert closest_nice_value(4.9, mode="upper") == 5.0
    assert closest_nice_value(4.9, mode="lower") == 4.5
    assert closest_nice_value(4.9e-2, mode="upper") == 5.0e-2
    assert closest_nice_value(4.9e2, mode="upper") == 5.0e2

    # Smaller increments below 2
    assert closest_nice_value(1.1e-2, mode="upper") == 1.25e-2
    assert closest_nice_value(1.1e2, mode="lower") == 1.0e2
    # Custom allowed
    assert closest_nice_value(0.9, allowed=[5.0], mode="lower") == 0.5


def test_nice_values():
    assert np.all(nice_values(0.0, 10.0, n=6) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert np.all(nice_values(-2.0, 2.0, n=5) == [-2.0, -1.0, 0.0, 1.0, 2.0])
    assert np.all(nice_values(1.19, 2.19, n=2) == [1.25, 2.0])
    assert np.all(nice_values(1.19e5, 2.19e5, n=2) == [1.25e5, 2.0e5])


def test_shortest_label(qtbot: QtBot):
    fm = QtGui.QFontMetrics(QtGui.QFont())
    assert shortest_label(fm, 1e0)[0] == "1"
    assert shortest_label(fm, 1e1)[0] == "10"
    assert shortest_label(fm, 1e2)[0] == "100"
    assert shortest_label(fm, 1e3)[0] == "1000"
    assert shortest_label(fm, 1e4)[0] == "10000"
    assert shortest_label(fm, 1e5)[0] == "100000"
    assert shortest_label(fm, 1e6)[0] == "1e+06"


def test_path_for_colorbar_labels(qtbot: QtBot):
    path = path_for_colorbar_labels(QtGui.QFont(), 0.0, 10.0, 100.0)
    assert path.boundingRect().width() > 90.0
    assert path.boundingRect().width() < 100.0
    path = path_for_colorbar_labels(QtGui.QFont(), 0.0, 10.0, 1000.0)
    assert path.boundingRect().width() > 900.0
    assert path.boundingRect().width() < 1000.0
