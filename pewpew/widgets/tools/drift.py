import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

from pewpew.charts.base import BaseChart
from pewpew.charts.colors import light_theme, sequential

from pewpew.lib.numpyqt import array_to_polygonf

from pewpew.graphics.items import ResizeableRectItem
from pewpew.graphics.imageitems import LaserImageItem
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.lasergraphicsview import LaserGraphicsView

from pewpew.widgets.views import TabView
from pewpew.widgets.tools.tool import ToolWidget

from typing import Any, Optional

# TODO replace drift chart with a SignalChart


class DriftChart(BaseChart):
    """Display the drift data and fit."""
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(QtCharts.QChart(), theme=light_theme, parent=parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        self.chart().legend().hide()

        self.xaxis = QtCharts.QValueAxis()
        self.xaxis.setLabelFormat("%d")
        self.yaxis = QtCharts.QValueAxis()
        self.yaxis.setVisible(False)

        self.chart().addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.chart().addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.drift1 = QtCharts.QLineSeries()
        self.drift1.setPen(QtGui.QPen(sequential[1], 1.0))
        self.chart().addSeries(self.drift1)
        self.drift1.attachAxis(self.xaxis)
        self.drift1.attachAxis(self.yaxis)

        self.drift2 = QtCharts.QLineSeries()
        self.drift2.setPen(QtGui.QPen(sequential[1], 1.0))
        self.chart().addSeries(self.drift2)
        self.drift2.attachAxis(self.xaxis)
        self.drift2.attachAxis(self.yaxis)

        self.fit = QtCharts.QSplineSeries()
        self.fit.setPen(QtGui.QPen(sequential[2], 2.0))
        self.chart().addSeries(self.fit)
        self.fit.attachAxis(self.xaxis)
        self.fit.attachAxis(self.yaxis)

    def drawDrift(self, x: np.ndarray, y: np.ndarray):
        if np.any(np.isnan(y)):
            nanstart = np.argmax(np.isnan(y))
            nanend = y.size - np.argmax(np.isnan(y[::-1]))

            points1 = np.stack((x[:nanstart], y[:nanstart]), axis=1)
            points2 = np.stack((x[nanend:], y[nanend:]), axis=1)

            poly1 = array_to_polygonf(points1)
            poly2 = array_to_polygonf(points2)

            self.drift1.replace(poly1)
            self.drift2.replace(poly2)
        else:
            points = np.stack((x, y), axis=1)
            poly = array_to_polygonf(points)

            self.drift1.replace(poly)
            self.drift2.clear()

        self.xaxis.setRange(0, x[-1])
        self.yaxis.setRange(np.nanmin(y), np.nanmax(y))

    def drawFit(self, x: np.ndarray, y: np.ndarray):
        points = np.stack((x, y), axis=1)
        poly = array_to_polygonf(points)

        self.fit.replace(poly)


class DriftGuideRectItem(ResizeableRectItem):
    """Rectangle for interactively selecting a drift region."""
    def __init__(
        self,
        rect: QtCore.QRectF,
        px: float,
        py: float,
        trim_enabled: bool = False,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(rect, parent=parent)

        pen = QtGui.QPen(QtCore.Qt.white, 2.0)
        pen.setCosmetic(True)
        self.setPen(pen)

        self.px = px
        self.py = py
        self.trim_enabled = trim_enabled

        self.top = rect.y() + rect.height() * 0.33
        self.top = self.top - self.top % py
        self.bottom = rect.y() + rect.height() * 0.66
        self.bottom = self.bottom - self.bottom % py

        self.changed = False

    def edgeAt(self, pos: QtCore.QPointF) -> Optional[str]:
        view = next(iter(self.scene().views()), None)
        if view is None:
            return None
        dist = (
            view.mapToScene(QtCore.QRect(0, 0, 10, 1))
            .boundingRect()
            .width()
        )

        if abs(self.rect().left() - pos.x()) < dist:
            return "left"
        elif abs(self.rect().right() - pos.x()) < dist:
            return "right"
        elif abs(pos.y() - self.top) < dist:
            return "top" if self.trim_enabled else None
        elif abs(pos.y() - self.bottom) < dist:
            return "bottom" if self.trim_enabled else None
        else:
            return None

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.selected_edge in ["top", "bottom"]:
            if self.selected_edge == "top" and event.pos().y() < self.bottom:
                self.top = event.pos().y() - event.pos().y() % self.py
            elif self.selected_edge == "bottom" and event.pos().y() > self.top:
                self.bottom = event.pos().y() - event.pos().y() % self.py
            self.changed = True
            self.update()
        else:
            super().mouseMoveEvent(event)

    def isSelected(self) -> bool:
        return True

    def itemChange(
        self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value: Any
    ) -> Any:
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            pos = QtCore.QPointF(value)
            pos.setX(pos.x() - pos.x() % self.px)
            pos.setY(self.rect().y())
            self.changed = True
            return pos
        return super().itemChange(change, value)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        option.state &= ~QtWidgets.QStyle.State_Selected
        super().paint(painter, option, widget)

        if self.trim_enabled:
            painter.drawLine(
                QtCore.QPointF(self.rect().left(), self.top),
                QtCore.QPointF(self.rect().right(), self.top),
            )
            painter.drawLine(
                QtCore.QPointF(self.rect().left(), self.bottom),
                QtCore.QPointF(self.rect().right(), self.bottom),
            )
            rect = self.rect()
            rect.setBottom(self.top)
            painter.fillRect(rect, QtGui.QBrush(QtGui.QColor(255, 255, 255, 32)))
            rect = self.rect()
            rect.setTop(self.bottom)
            painter.fillRect(rect, QtGui.QBrush(QtGui.QColor(255, 255, 255, 32)))
        else:
            painter.fillRect(self.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 32)))


class DriftGraphicsView(LaserGraphicsView):
    """Graphics view with drift selection and display."""
    driftChanged = QtCore.Signal()

    def __init__(self, options: GraphicsOptions, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(options, parent=parent)
        self.setInteractionFlag("tool")

        self.guide: Optional[DriftGuideRectItem] = None

    def drawGuides(self) -> None:
        trim = False
        if self.guide is not None:
            trim = self.guide.trim_enabled
            self.scene().removeItem(self.guide)

        px = self.image.rect.width() / self.data.shape[1]
        py = self.image.rect.height() / self.data.shape[0]

        x1 = self.image.rect.x() + 0.1 * self.image.rect.width()
        x1 = x1 - x1 % px
        x2 = self.image.rect.x() + 0.2 * self.image.rect.width()
        x2 = x2 - x2 % px

        rect = QtCore.QRectF(x1, self.image.rect.y(), x2 - x1, self.image.rect.height())

        self.guide = DriftGuideRectItem(rect, px, py, trim_enabled=trim)
        self.guide.setZValue(self.image.zValue() + 1)
        self.scene().addItem(self.guide)

    def driftData(self) -> Optional[np.ndarray]:
        if self.guide is None:
            return None
        rect = self.guide.rect()
        rect.setTop(self.guide.top)
        rect.setBottom(self.guide.bottom)

        p1 = self.mapToData(self.guide.mapToScene(rect.topLeft()))
        p2 = self.mapToData(self.guide.mapToScene(rect.bottomRight()))

        x1, y1 = max(p1.x(), 0), max(p1.y(), 0)
        x2, y2 = min(p2.x(), self.data.shape[1]), min(p2.y(), self.data.shape[0])

        drift = self.data[:, x1:x2].copy()

        if self.guide.trim_enabled:
            drift[y1:y2] = np.nan

        return drift

    def setTrim(self, trim: bool) -> None:
        self.guide.trim_enabled = trim
        self.guide.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        if self.guide is not None and self.guide.changed:
            self.guide.changed = False
            self.driftChanged.emit()


class DriftTool(ToolWidget):
    """Drift normalisation tool."""
    normalise_methods = ["Maximum", "Minimum"]

    def __init__(self, item: LaserImageItem, view: TabView):
        super().__init__(item, apply_all=False, view=view)

        self.drift: Optional[np.ndarray] = None

        self.graphics = DriftGraphicsView(self.viewspace.options, parent=self)
        self.graphics.driftChanged.connect(self.updateDrift)
        self.graphics.driftChanged.connect(self.updateNormalise)

        self.chart = DriftChart(parent=self)

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.activated.connect(self.refresh)

        self.spinbox_degree = QtWidgets.QSpinBox()
        self.spinbox_degree.setRange(0, 9)
        self.spinbox_degree.setValue(3)
        self.spinbox_degree.valueChanged.connect(self.updateDrift)
        self.spinbox_degree.setToolTip(
            "Degree of polynomial used to fit the drift,\nuse 0 for the raw data."
        )

        self.combo_normalise = QtWidgets.QComboBox()
        self.combo_normalise.addItems(DriftTool.normalise_methods)
        self.combo_normalise.setCurrentText("Minimum")
        self.combo_normalise.activated.connect(self.updateNormalise)

        self.lineedit_normalise = QtWidgets.QLineEdit()
        self.lineedit_normalise.setEnabled(False)

        self.check_trim = QtWidgets.QCheckBox("Show drift trim controls.")
        self.check_trim.toggled.connect(self.graphics.setTrim)
        self.check_trim.toggled.connect(self.updateDrift)

        self.check_apply_all = QtWidgets.QCheckBox("Apply to all elements.")

        layout_graphics = QtWidgets.QVBoxLayout()
        layout_graphics.addWidget(self.graphics, 2)
        layout_graphics.addWidget(self.chart, 1)
        layout_graphics.addWidget(self.combo_element, 0, QtCore.Qt.AlignRight)
        self.box_graphics.setLayout(layout_graphics)

        layout_norm = QtWidgets.QVBoxLayout()
        layout_norm.addWidget(self.combo_normalise)
        layout_norm.addWidget(self.lineedit_normalise)

        layout_controls = QtWidgets.QFormLayout()
        layout_controls.addRow("Degree of fit:", self.spinbox_degree)
        layout_controls.addRow("Normalise to:", layout_norm)
        layout_controls.addRow(self.check_trim)
        layout_controls.addRow(self.check_apply_all)
        self.box_controls.setLayout(layout_controls)

        self.initialise()

    def apply(self) -> None:
        if self.drift is None:
            return
        if self.combo_normalise.currentText() == "Maximum":
            value = np.amax(self.drift)
        elif self.combo_normalise.currentText() == "Minimum":
            value = np.amin(self.drift)
        else:
            raise ValueError("Unknown normalisation method.")

        if self.check_apply_all.isChecked():
            names = self.widget.laser.elements
        else:
            names = [self.combo_element.currentText()]

        for name in names:
            transpose = self.widget.laser.data[name].T
            transpose /= self.drift / value

        self.refresh()

    def initialise(self) -> None:
        elements = self.widget.laser.elements
        self.combo_element.clear()
        self.combo_element.addItems(elements)

        self.refresh()

    def isComplete(self) -> bool:
        return self.drift is not None

    def updateDrift(self) -> None:
        ys = self.graphics.driftData()
        ys = np.nanmean(ys, axis=1)

        xs = np.arange(ys.size)

        data = self.graphics.driftData()
        ys = np.nanmean(data, axis=1)
        xs = np.arange(ys.size)
        nans = np.isnan(ys)

        self.chart.drawDrift(xs, ys)

        deg = self.spinbox_degree.value()
        if deg == 0:
            self.drift = ys
        else:
            coef = np.polynomial.polynomial.polyfit(xs[~nans], ys[~nans], deg)
            self.drift = np.polynomial.polynomial.polyval(xs, coef)

        if self.drift is not None:
            self.chart.drawFit(xs, self.drift)

    def updateNormalise(self) -> None:
        if self.drift is None:
            return
        if self.combo_normalise.currentText() == "Maximum":
            value = np.amax(self.drift)
        elif self.combo_normalise.currentText() == "Minimum":
            value = np.amin(self.drift)
        else:
            raise ValueError("Unknown normalisation method.")
        self.lineedit_normalise.setText(f"{value:.8g}")

    def refresh(self) -> None:
        element = self.combo_element.currentText()

        data = self.widget.laser.get(element, flat=True, calibrated=False)

        x0, x1, y0, y1 = self.widget.laser.config.data_extent(data.shape)
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

        self.graphics.drawImage(data, rect, element)
        if self.graphics.guide is None:
            self.graphics.drawGuides()
        self.graphics.label.setText(element)

        self.graphics.setOverlayItemVisibility()
        self.graphics.updateForeground()
        self.graphics.invalidateScene()

        self.updateDrift()
        self.updateNormalise()
