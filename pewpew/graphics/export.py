from typing import List, Tuple

import numpy as np
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui

from pewpew.graphics import colortable
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.overlayitems import MetricScaleBarOverlay
from pewpew.graphics.util import path_for_colorbar_labels
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


def paint_colorbar(
    painter: QtGui.QPainter,
    rect: QtCore.QRectF,
    table: List[int],
    vrange: Tuple[float, float],
    unit: str | None = None,
) -> None:
    painter.save()

    _data = np.arange(256, dtype=np.uint8)
    _data[0] = 1
    cbar = QtGui.QImage(_data, 256, 1, 256, QtGui.QImage.Format_Indexed8)
    cbar.setColorTable(table)
    cbar.setColorCount(len(table))

    painter.drawImage(rect, cbar)

    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(QtCore.Qt.black, 2.0)
    pen.setCosmetic(True)
    painter.setPen(pen)
    painter.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.white))

    fm = painter.fontMetrics()

    path = path_for_colorbar_labels(painter.font(), vrange[0], vrange[1], rect.width())
    if unit is not None and len(unit) > 0:
        xpos = rect.width() - (
            fm.boundingRect(unit).width() + fm.lineWidth() + fm.rightBearing(unit[-1])
        )
        path.addText(xpos, fm.ascent() + fm.height(), painter.font(), unit)
    path.translate(rect.bottomLeft())

    painter.strokePath(path, painter.pen())
    painter.fillPath(path, painter.brush())
    painter.restore()


def paint_color_venn(
    painter: QtGui.QPainter,
    parent_rect: QtCore.QRectF,
    alignment: QtCore.Qt.AlignmentFlag,
    colors: List[QtGui.QColor],
) -> None:
    radius = painter.fontMetrics().xHeight() * 2.0
    rect = position_for_alignment(
        parent_rect, QtCore.QRectF(0, 0, radius * 3, radius * 3), alignment
    )
    centers = [
        QtCore.QPointF(rect.center().x(), rect.top() + radius),
        QtCore.QPointF(rect.left() + radius, rect.bottom() - radius),
        QtCore.QPointF(rect.right() - radius, rect.bottom() - radius),
    ]
    # Clear with black
    painter.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.black))
    for center in centers:
        painter.drawEllipse(center, radius, radius)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
    for color, center in zip(colors, centers):
        painter.setBrush(QtGui.QBrush(color))
        painter.drawEllipse(center, radius, radius)


def paint_scalebar(
    painter: QtGui.QPainter,
    rect: QtCore.QRectF,
    parent_rect: QtCore.QRectF,
    alignment: QtCore.Qt.AlignmentFlag,
    scale: float,
) -> None:
    painter.save()
    pen = QtGui.QPen(QtCore.Qt.black, 2.0)
    pen.setCosmetic(True)
    painter.setPen(pen)
    painter.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.white))
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

    rect = position_for_alignment(parent_rect, rect, alignment)

    width, unit = MetricScaleBarOverlay.getWidthAndUnit(rect.width() * scale, "μm")
    text = f"{width * 1e-6 / MetricScaleBarOverlay.units[unit]:.3g} {unit}"
    width = width / scale

    fm = painter.fontMetrics()
    path = QtGui.QPainterPath()
    path.addText(
        rect.center().x() - fm.boundingRect(text).width() / 2.0,
        rect.top() + fm.ascent(),
        painter.font(),
        text,
    )

    painter.strokePath(path, painter.pen())
    painter.fillPath(path, painter.brush())

    # Draw the bar
    bar = QtCore.QRectF(
        rect.center().x() - width / 2.0,
        rect.top() + fm.height(),
        width,
        fm.xHeight() / 2.0,
    )
    painter.drawRect(bar)
    painter.restore()


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
        return image

    fm = QtGui.QFontMetrics(options.font)
    xh = fm.xHeight()
    image = image.scaled(image.size() * 2.0)
    size = image.size()

    if colorbar:  # make room for colorbar
        size = size.grownBy(QtCore.QMargins(0, 0, 0, xh + xh / 2.0 + fm.height()))
        unit = laser.calibration[element].unit
        if unit is not None and len(unit) > 0:
            size = size.grownBy(QtCore.QMargins(0, 0, 0, fm.height()))

    pixmap = QtGui.QPixmap(size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)

    painter = QtGui.QPainter(pixmap)
    painter.setFont(options.font)

    # Draw the image
    painter.drawImage(image.rect(), image, image.rect())

    if colorbar:
        paint_colorbar(
            painter,
            QtCore.QRectF(
                image.rect().left(), image.rect().bottom() + xh / 2.0, image.width(), xh
            ),
            table,
            (vmin, vmax),
            unit=unit,
        )

    # Draw the element label
    if label_alignment is not None:
        rect = painter.boundingRect(
            image.rect().adjusted(xh, xh, -xh, -xh), label_alignment, element
        )
        path = QtGui.QPainterPath()
        path.addText(rect.left(), rect.top() + fm.ascent(), painter.font(), element)

        pen = QtGui.QPen(QtCore.Qt.black, 2.0)
        pen.setCosmetic(True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.strokePath(path, pen)
        painter.fillPath(path, QtGui.QBrush(QtCore.Qt.GlobalColor.white))

    # Draw the scale-bar
    if scalebar_alignment is not None:
        x0, x1, y0, y1 = laser.extent
        scale = (x1 - x0) / image.rect().width()
        paint_scalebar(
            painter,
            QtCore.QRectF(0, 0, xh * 10.0, fm.height()),
            image.rect().adjusted(xh, xh, -xh, -xh),
            scalebar_alignment,
            scale,
        )

    painter.end()
    return pixmap.toImage()


def generate_rgb_laser_image(
    laser: Laser,
    elements: List[str],
    colors: List[QtGui.QColor],
    ranges: List[Tuple[float, float]],
    options: GraphicsOptions,
    scalebar_alignment: QtCore.Qt.AlignmentFlag
    | None = QtCore.Qt.AlignmentFlag.AlignTop
    | QtCore.Qt.AlignmentFlag.AlignRight,
    label_alignment: QtCore.Qt.AlignmentFlag
    | None = QtCore.Qt.AlignmentFlag.AlignTop
    | QtCore.Qt.AlignmentFlag.AlignLeft,
    venn_alignment: QtCore.Qt.AlignmentFlag
    | None = QtCore.Qt.AlignmentFlag.AlignTop
    | QtCore.Qt.AlignmentFlag.AlignLeft,
    raw: bool = False,
    subtractive: bool = False,
) -> QtGui.QImage:
    data = np.zeros((*laser.shape[:2], 3))
    for i, (element, color, (pmin, pmax)) in enumerate(zip(elements, colors, ranges)):
        if element not in laser.elements:
            continue
        rgb = np.array(color.getRgbF()[:3])
        if subtractive:
            rgb = 255.0 - rgb

        # Normalise to range
        x = laser.get(element=element, calibrate=False, flat=True)
        vmin, vmax = np.percentile(x, (pmin, pmax))
        x = np.clip(x, vmin, vmax)
        if vmin != vmax:
            x = (x - vmin) / (vmax - vmin)
        # Convert to separate rgb channels
        data += x[:, :, None] * rgb

    if subtractive:
        data = np.full_like(data, 255) - data

    image = array_to_image(data)

    if raw:
        return image

    fm = QtGui.QFontMetrics(options.font)
    xh = fm.xHeight()
    image = image.scaled(image.size() * 2.0)
    size = image.size()

    pixmap = QtGui.QPixmap(size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)

    painter = QtGui.QPainter(pixmap)
    painter.setFont(options.font)

    # Draw the image
    painter.drawImage(image.rect(), image, image.rect())

    # Draw the element label
    if label_alignment is not None:
        width = max(fm.boundingRect(text).width() for text in elements)
        rect = QtCore.QRectF(0, 0, width, fm.height() * len(elements))

        rect = position_for_alignment(
            image.rect().adjusted(xh, xh, -xh, -xh), rect, label_alignment
        )

        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(QtCore.Qt.black, 2.0)
        pen.setCosmetic(True)

        pos = rect.topLeft()
        for element, color in zip(elements, colors):
            path = QtGui.QPainterPath()
            path.addText(pos.x(), pos.y() + fm.ascent(), painter.font(), element)
            painter.strokePath(path, pen)
            painter.fillPath(path, QtGui.QBrush(color))
            pos.setY(pos.y() + fm.height())

    # Draw the scale-bar
    if scalebar_alignment is not None:
        x0, x1, y0, y1 = laser.extent
        scale = (x1 - x0) / image.rect().width()
        paint_scalebar(
            painter,
            QtCore.QRectF(0, 0, xh * 10.0, fm.height()),
            image.rect().adjusted(xh, xh, -xh, -xh),
            scalebar_alignment,
            scale,
        )

    # Draw the color Venn
    if venn_alignment is not None:
        paint_color_venn(
            painter, image.rect().adjusted(xh, xh, -xh, -xh), venn_alignment, colors
        )

    painter.end()
    return pixmap.toImage()
