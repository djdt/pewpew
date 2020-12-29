from PySide2 import QtGui

import ctypes
import numpy as np
import shiboken2

import pewpew.lib.polyext


def array_as_indexed8(
    array: np.ndarray, vmin: float = None, vmax: float = None
) -> np.ndarray:
    if vmin is None:
        vmin = np.amin(array)
    if vmax is None:
        vmax = np.amax(array)
    if vmin > vmax:
        vmin, vmax = vmax, vmin

    array = np.atleast_2d(array)
    data = np.clip(array, vmin, vmax)
    data = (data - vmin) / (vmax - vmin)
    return (data * 255.0).astype(np.uint8)


def polygonf_to_array(polygon: QtGui.QPolygonF) -> np.ndarray:
    buf = (ctypes.c_double * 2 * polygon.length()).from_address(
        shiboken2.getCppPointer(polygon.data())[0]
    )
    return np.frombuffer(buf, dtype=np.float64).reshape(-1, 2)


def polygon_contains_points(polygon: QtGui.QPolygonF, points: np.ndarray) -> np.ndarray:
    poly_array = polygonf_to_array(polygon)
    result = pewpew.lib.polyext.polygon_contains_points(poly_array, points)
    return result
