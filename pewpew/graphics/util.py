from PySide6 import QtGui

import numpy as np

from pewpew.lib.numpyqt import polygonf_to_array
import pewpew.lib.polyext


def polygonf_contains_points(
    polygon: QtGui.QPolygonF, points: np.ndarray
) -> np.ndarray:
    """Check if a any points are contained within a polygon."""
    poly_array = polygonf_to_array(polygon)
    result = pewpew.lib.polyext.polygonf_contains_points(poly_array, points)
    return result
