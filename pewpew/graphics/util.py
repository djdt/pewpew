
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
    mode: str = "lower",
) -> np.ndarray:
    values = np.asarray(values)
    if allowed is None:
        allowed = np.concatenate([np.arange(0, 2.0, 0.25), np.arange(2, 10, 0.5)])
    else:
        allowed = np.asarray(allowed)

    with np.errstate(divide="ignore"):
        e = np.floor(np.log10(np.abs(values)))
    nice = np.divide(values, 10**e, where=values != 0.0)

    if mode == "upper":
        idx = np.searchsorted(allowed, np.abs(nice), side="left")
    elif mode == "lower":
        idx = np.searchsorted(allowed, np.abs(nice), side="right") - 1
    else:
        raise ValueError("'mode' must be one of 'upper', 'lower'")
    idx = np.clip(idx, 0, allowed.size - 1)
    return allowed[idx] * (10**e) * np.sign(values)


def nice_values(vmin: float, vmax: float, n: int = 6) -> np.ndarray:
    lower = closest_nice_value(vmin, mode="upper")
    upper = closest_nice_value(vmax, mode="lower")
    if n == 2:
        return np.array([lower, upper])
    interval = closest_nice_value(
        (upper - lower) / (n - 1),
        allowed=np.array([0.0, 1.0, 2.0, 2.5, 5.0, 7.5]),
        mode="upper",
    )
    return np.arange(lower, vmax * 1.001, interval)


def shortest_label(
    fm: QtGui.QFontMetrics, value: float, prec: int = 2
) -> tuple[str, float]:
    g_label = f"{value:{prec}g}".strip()
    g_width = fm.boundingRect(g_label).width()
    if value < 10**prec:
        return g_label, g_width

    d_label = f"{int(value):{prec}d}".strip()
    d_width = fm.boundingRect(d_label).width()
    if g_width < d_width:
        return g_label, g_width
    else:
        return d_label, d_width


def path_for_colorbar_labels(
    font: QtGui.QFont, vmin: float, vmax: float, width: float, pen_width: float = 1.0
) -> QtGui.QPainterPath:
    vrange = vmax - vmin
    fm = QtGui.QFontMetrics(font)
    path = QtGui.QPainterPath()

    check_width = fm.xHeight() / 4.0

    for n in np.arange(6, 1, -1):
        max_text_width = shortest_label(fm, 8.8e88)[1]
        if max_text_width * n < width:
            break

    values = nice_values(vmin, vmax, n)
    for i, v in enumerate(values):
        text, text_width = shortest_label(fm, v)
        xpos = v * width / vrange
        text_pos = xpos - text_width / 2.0
        if text_pos < 0.0:
            text_pos = fm.lineWidth()
        elif text_pos + text_width > width:
            text_pos = width - text_width - fm.lineWidth()
        path.addText(text_pos, fm.ascent(), font, text)

        xpos -= check_width / 2.0
        if xpos < 0.0:
            xpos = pen_width / 2.0
        elif xpos + check_width > width:
            xpos = width - check_width - pen_width / 2.0
        path.addRect(xpos, -check_width, check_width, check_width * 2.0)

    return path
