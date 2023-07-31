import numpy as np
from PySide6 import QtCore, QtGui

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


def shortest_label(fm: QtGui.QFontMetrics, value: float, prec: int = 2) -> str:
    g_label = f"{value:{prec}g}".strip()
    if value < 10**prec:
        return g_label
    d_label = f"{int(value):{prec}d}".strip()
    if fm.boundingRect(g_label).width() < fm.boundingRect(d_label).width():
        return g_label
    else:
        return d_label


def paint_colorbar_labels(
    painter: QtGui.QPainter, vmin: float, vmax: float, rect: QtCore.QRectF
) -> None:
    fm = painter.fontMetrics()
    vrange = vmax - vmin

    path = QtGui.QPainterPath()

    # Label for vmin, centered if able else left aligned with end
    fmin = float(closest_nice_value(vmin, mode="upper"))
    text = shortest_label(fm, fmin)
    fmin = fmin * rect.width() / vrange
    xmin = fmin - fm.boundingRect(text).width() / 2.0
    if xmin < rect.left():
        xmin = rect.left() + fm.lineWidth() + fm.leftBearing(text[0])

    path.addText(xmin, rect.top(), painter.font(), text)

    # Label for vmax, centered if able else right aligned with end
    fmax = float(closest_nice_value(vmax, mode="upper"))
    text = shortest_label(fm, fmax)
    fmax = fmax * rect.width() / vrange
    width = fm.boundingRect(text).width()
    xmax = fmax - width / 2.0
    if xmax + width > rect.right():
        xmax = rect.lright() - width - fm.lineWidth() - fm.rightBearing(text[-1])

    path.addText(xmax, rect.top(), painter.font(), text)

    # Other labels
    for nlabels in [7, 5, 3, 1]:
        vals = np.linspace(vmin, max, nlabels + 2)[1:-1]
        fvals = closest_nice_value(vals, mode="closest")
        texts = [shortest_label(fm, fv) for fv in fvals]
        twidth = sum(fm.boundingRect(text).widths() for text in texts)
        if twidth > 
