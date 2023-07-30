import numpy as np
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui

from pewpew.graphics import colortable
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.overlayitems import MetricScaleBarOverlay
from pewpew.graphics.util import closest_nice_value
from pewpew.lib.numpyqt import array_to_image


def position_for_alignment(
    bounds: QtCore.QRectF, rect: QtCore.QRectF, alignment: QtCore.Qt.AlignmentFlag
) -> QtCore.QPointF:
    rect.moveTopLeft(bounds.topLeft())
    if alignment & QtCore.Qt.AlignRight:
        rect.moveRight(bounds.right())
    elif alignment & QtCore.Qt.AlignHCenter:
        rect.moveCenter(QtCore.QPointF(bounds.center().x(), rect.center().y()))
    if alignment & QtCore.Qt.AlignBottom:
        rect.moveBottom(bounds.bottom())
    elif alignment & QtCore.Qt.AlignVCenter:
        rect.moveCenter(QtCore.QPointF(rect.center().x(), bounds.center().y()))
    return rect


def shortest_label(fm: QtGui.QFontMetrics, value: float, prec: int = 2) -> str:
    g_label = f"{value:{prec}g}".strip()
    if value < 10**prec:
        return g_label
    d_label = f"{int(value):{prec}d}".strip()
    if fm.boundingRect(g_label).width() < fm.boundingRect(d_label).width():
        return g_label
    else:
        return d_label


def generate_laser_image(
    laser: Laser,
    element: str,
    options: GraphicsOptions,
    scalebar_alignment: QtCore.Qt.AlignmentFlag
    | None = QtCore.Qt.AlignmentFlag.AlignTop
    | QtCore.Qt.AlignmentFlag.AlignRight,
    label_alignment: QtCore.Qt.AlignmentFlag
    | None = QtCore.Qt.AlignmentFlag.AlignTop
    | QtCore.Qt.AlignmentFlag.AlignLeft,
    colorbar: bool = True,
    raw: bool = False,
) -> QtGui.QImage:
    data = laser.get(element, calibrate=options.calibrate, flat=True)
    data = np.ascontiguousarray(data)

    vmin, vmax = options.get_color_range_as_float(element, data)
    table = colortable.get_table(options.colortable)

    data = np.clip(data, vmin, vmax)
    if vmin != vmax:  # Avoid div 0
        data = (data - vmin) / (vmax - vmin)

    image = array_to_image(data)
    image.setColorTable(table)
    image.setColorCount(len(table))

    if raw:
        return laser

    bar_width = 10.0
    pad = 10.0
    image = image.scaled(image.size() * 2.0)
    size = image.size()

    if colorbar:  # make room for colorbar
        size = size.grownBy(
            QtCore.QMargins(
                0,
                0,
                0,
                bar_width + pad / 2.0 + QtGui.QFontMetrics(options.font).height(),
            )
        )
    colorbar_unit = (
        colorbar and options.calibrate and laser.calibration[element].unit is not None
    )
    if colorbar_unit:
        size = size.grownBy(
            QtCore.QMargins(0, 0, 0, 5 + QtGui.QFontMetrics(options.font).height())
        )

    pixmap = QtGui.QPixmap(size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)

    pen = QtGui.QPen(QtCore.Qt.black, 2.0)
    pen.setCosmetic(True)

    painter = QtGui.QPainter(pixmap)
    painter.setFont(options.font)
    fm = painter.fontMetrics()
    # Draw the image
    painter.drawImage(image.rect(), image, image.rect())

    if colorbar:
        rect = QtCore.QRectF(
            image.rect().left(),
            image.rect().bottom() + pad / 2.0,
            image.width(),
            bar_width,
        )
        _data = np.arange(256, dtype=np.uint8)
        cbar = QtGui.QImage(_data, 256, 1, 256, QtGui.QImage.Format_Indexed8)
        cbar.setColorTable(table)
        cbar.setColorCount(len(table))

        painter.drawImage(rect, cbar)

        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Labels and checks
        nmin = closest_nice_value(vmin, mode="upper")
        nmid = closest_nice_value((vmax + vmin) / 2.0, mode="closest")
        nmax = closest_nice_value(vmax, mode="lower")

        check_pos_min = rect.width() / (vmax - vmin) * nmin
        check_pos_mid = rect.width() / (vmax - vmin) * nmid
        check_pos_max = rect.width() / (vmax - vmin) * nmax

        path = QtGui.QPainterPath()
        path.addRect(check_pos_min + 1, rect.bottom() - 2, 2, 4)
        path.addRect(check_pos_mid, rect.bottom() - 2, 2, 4)
        path.addRect(check_pos_max - 1, rect.bottom() - 2, 2, 4)

        painter.strokePath(path, pen)
        painter.fillPath(path, QtGui.QBrush(QtCore.Qt.GlobalColor.white))

        path = QtGui.QPainterPath()
        # Left label
        text = shortest_label(fm, nmin, 2)
        xpos = check_pos_min - fm.boundingRect(text).width() / 2.0
        if xpos < rect.left():
            xpos = rect.left() + fm.lineWidth() + fm.leftBearing(text[0])
        path.addText(xpos, rect.bottom() + fm.ascent(), painter.font(), text)
        # Right label
        text = shortest_label(fm, nmax, 2)
        width = fm.boundingRect(text).width()
        xpos = check_pos_max - width / 2.0
        if xpos + width > rect.right():
            xpos = rect.right() - width - fm.lineWidth() - fm.rightBearing(text[-1])
        path.addText(xpos, rect.bottom() + fm.ascent(), painter.font(), text)
        # Center label
        text = shortest_label(fm, nmid, 2)
        xpos = check_pos_mid - fm.boundingRect(text).width() / 2.0
        path.addText(xpos, rect.bottom() + fm.ascent(), painter.font(), text)
        # unit
        if colorbar_unit:
            path.addText(
                rect.right() - fm.boundingRect(laser.calibration[element].unit).width(),
                rect.bottom() + fm.ascent() + fm.height(),
                painter.font(),
                laser.calibration[element].unit,
            )
        painter.strokePath(path, pen)
        painter.fillPath(path, QtGui.QBrush(QtCore.Qt.GlobalColor.white))

    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    # Draw the element label
    if label_alignment is not None:
        rect = painter.boundingRect(
            image.rect().adjusted(pad, pad, -pad, -pad), label_alignment, element
        )
        path = QtGui.QPainterPath()
        path.addText(rect.left(), rect.top() + fm.ascent(), painter.font(), element)
        painter.strokePath(path, pen)
        painter.fillPath(path, QtGui.QBrush(QtCore.Qt.GlobalColor.white))

    # Draw the scale-bar
    if scalebar_alignment is not None:
        rect = QtCore.QRectF(0, 0, 100.0, fm.height())
        rect = position_for_alignment(
            image.rect().adjusted(pad, pad, -pad, -pad), rect, scalebar_alignment
        )

        x0, x1, y0, y1 = laser.extent
        scale = (x1 - x0) / image.rect().width()

        width, unit = MetricScaleBarOverlay.getWidthAndUnit(rect.width() * scale, "μm")
        text = f"{width * 1e-6 / MetricScaleBarOverlay.units[unit]:.3g} {unit}"
        width = width / scale

        path = QtGui.QPainterPath()
        path.addText(
            rect.center().x() - fm.boundingRect(text).width() / 2.0,
            rect.top() + fm.ascent(),
            painter.font(),
            text,
        )

        painter.strokePath(path, pen)
        painter.fillPath(path, QtGui.QBrush(QtCore.Qt.GlobalColor.white))

        # Draw the bar
        bar = QtCore.QRectF(
            rect.center().x() - width / 2.0, rect.top() + fm.height(), width, 5
        )
        painter.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.white))
        painter.drawRect(bar)

    painter.end()
    return pixmap.toImage()
