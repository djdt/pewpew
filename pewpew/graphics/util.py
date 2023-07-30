import numpy as np
from PySide6 import QtGui

import pewpew.lib.polyext
from pewpew.lib.numpyqt import polygonf_to_array


def polygonf_contains_points(
    polygon: QtGui.QPolygonF, points: np.ndarray
) -> np.ndarray:
    """Check if a any points are contained within a polygon."""
    poly_array = polygonf_to_array(polygon)
    result = pewpew.lib.polyext.polygonf_contains_points(poly_array, points)
    return result


def closest_nice_value(
    values: float | np.ndarray,
    allowed: np.ndarray | None = None,
    mode: str = "closest",
) -> np.ndarray:
    values = np.asarray(values)
    if allowed is None:
        allowed = np.array(
            [
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
                1.25,
                1.5,
                2.0,
                2.5,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                7.5,
                8.0,
                9.0,
            ]
        )
    allowed = np.asarray(allowed)

    pwrs = 10 ** (np.where(np.abs(values) != 0.0, np.log10(np.abs(values)), 0)).astype(
        int
    )
    nice = values / pwrs

    upper_idx = np.searchsorted(allowed, nice, side="left")
    upper_idx = np.clip(upper_idx, 0, allowed.size - 1)
    upper_nice = allowed[upper_idx] * pwrs

    lower_idx = np.searchsorted(allowed, nice, side="right") - 1
    lower_idx = np.clip(lower_idx, 0, allowed.size - 1)
    lower_nice = allowed[lower_idx] * pwrs

    if mode == "upper":
        return upper_nice
    elif mode == "lower":
        return lower_nice
    elif mode == "closest":
        mask = np.abs(upper_nice - values) < np.abs(lower_nice - values)
        return np.where(mask, upper_nice, lower_nice)
    else:
        raise ValueError("'mode' must be one of 'upper', 'lower', 'closest'")
