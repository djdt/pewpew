from PySide6 import QtCore, QtGui
from pewpew.graphics.util import polygonf_contains_points

import numpy as np


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
