import copy
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

from pewlib import Calibration

from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.items import ResizeableRectItem
from pewpew.graphics.options import GraphicsOptions

from pewpew.validators import DoubleSignificantFiguresDelegate

from pewpew.widgets.dialogs import CalibrationCurveDialog
from pewpew.models import CalibrationPointsTableModel
from pewpew.widgets.modelviews import (
    BasicTable,
    BasicTableView,
)
from pewpew.widgets.laser import LaserWidget

from .tool import ToolWidget

from typing import Any, Dict, List, Optional, Tuple


class StandardsGraphicsView(LaserGraphicsView):
    """Graphics view with rectangle selection of areas."""

    levelsChanged = QtCore.Signal()

    def __init__(self, options: GraphicsOptions, parent: QtWidgets.QWidget = None):
        super().__init__(options, parent)
        self.setInteractionFlag("tool")
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)

        self.levels: List[CalibrationRectItem] = []

    def drawLevels(self, labels: str, n: int) -> None:
        for item in self.levels:
            self.scene().removeItem(item)
        self.levels = []

        px = self.image.rect.width() / self.data.shape[1]
        py = self.image.rect.height() / self.data.shape[0]

        x1 = self.image.rect.x() + 0.1 * self.image.rect.width()
        x1 = x1 - x1 % px
        x2 = self.image.rect.x() + 0.9 * self.image.rect.width()
        x2 = x2 - x2 % px

        height_per_level = self.image.rect.height() / n

        for i in range(n):
            y1 = self.image.rect.y() + i * height_per_level
            y1 = y1 - y1 % py
            y2 = self.image.rect.y() + (i + 1) * height_per_level
            y2 = y2 - y2 % py

            rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

            item = CalibrationRectItem(rect, labels[i], px, py, font=self.options.font)
            item.setZValue(self.image.zValue() + 1)
            self.levels.append(item)
            self.scene().addItem(item)

    def currentLevelDataCoords(self) -> List[Tuple[int, int, int, int]]:
        levels = []
        for item in self.levels:
            p1 = self.mapToData(item.mapToScene(item.rect().topLeft()))
            p2 = self.mapToData(item.mapToScene(item.rect().bottomRight()))
            levels.append((p1.x(), p1.y(), p2.x(), p2.y()))
        return levels

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        for item in self.levels:
            if item.changed:
                self.levelsChanged.emit()
                break
        for item in self.levels:
            item.changed = False
        super().mouseReleaseEvent(event)


class StandardsTool(ToolWidget):
    """Tool for creating calibrations from a laser image."""

    WEIGHTINGS = ["Equal", "1/σ²", "x", "1/x", "1/x²", "y", "1/y", "1/y²"]

    def __init__(self, widget: LaserWidget):
        super().__init__(widget, control_label="Calibration", apply_all=True)
        self.setWindowTitle("Calibration Tool")

        self.calibration: Dict[str, Calibration] = {}
        self.previous_element = ""
        self.dlg: Optional[CalibrationCurveDialog] = None

        # Left side
        self.spinbox_levels = QtWidgets.QSpinBox()
        self.spinbox_levels.setMinimum(1)
        self.spinbox_levels.setMaximum(20)
        self.spinbox_levels.setValue(6)
        self.spinbox_levels.valueChanged.connect(self.spinBoxLevels)

        self.lineedit_units = QtWidgets.QLineEdit()
        self.lineedit_units.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        self.lineedit_units.editingFinished.connect(self.lineeditUnits)

        self.combo_weighting = QtWidgets.QComboBox()
        self.combo_weighting.addItems(self.WEIGHTINGS)
        self.combo_weighting.currentIndexChanged.connect(self.comboWeighting)

        self.results = StandardsResultsTable(self)
        self.results.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow
        )
        self.results.horizontalHeader().setStretchLastSection(True)
        self.button_plot = QtWidgets.QPushButton("Plot")
        self.button_plot.pressed.connect(self.showCurve)

        # Right side
        self.graphics = StandardsGraphicsView(self.viewspace.options, parent=self)
        self.graphics.levelsChanged.connect(self.updateCounts)

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.currentIndexChanged.connect(self.comboElement)

        self.table = StandardsTable(Calibration(), self)
        self.table.model().setRowCount(6)
        self.table.model().dataChanged.connect(self.completeChanged)
        self.table.model().dataChanged.connect(self.updateResults)

        # Initialise
        self.calibration = copy.deepcopy(self.widget.laser.calibration)
        # Prevent currentIndexChanged being emmited
        self.combo_element.blockSignals(True)
        self.combo_element.clear()
        self.combo_element.addItems(self.widget.laser.elements)
        self.combo_element.setCurrentText(self.widget.combo_element.currentText())
        self.combo_element.blockSignals(False)

        element = self.combo_element.currentText()
        self.combo_weighting.setCurrentText(self.calibration[element].weighting)
        self.lineedit_units.setText(self.calibration[element].unit)
        self.table.model().setCalibration(self.calibration[element], resize=False)

        layout_cal_form = QtWidgets.QFormLayout()
        layout_cal_form.addRow("Levels:", self.spinbox_levels)

        layout_table_form = QtWidgets.QFormLayout()
        layout_table_form.addRow("Units:", self.lineedit_units)
        layout_table_form.addRow("Weighting:", self.combo_weighting)

        layout_results = QtWidgets.QHBoxLayout()
        layout_results.addWidget(self.results, 1)
        layout_results.addWidget(
            self.button_plot, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignTop
        )
        layout_results.addWidget(self.combo_element, 0, QtCore.Qt.AlignTop)

        layout_graphics = QtWidgets.QVBoxLayout()
        layout_graphics.addWidget(self.graphics, 1)
        layout_graphics.addLayout(layout_results)
        self.box_graphics.setLayout(layout_graphics)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addLayout(layout_cal_form)
        layout_controls.addWidget(self.table)
        layout_controls.addLayout(layout_table_form)
        self.box_controls.setLayout(layout_controls)

        self.refresh()
        self.updateResults()

    def apply(self) -> None:
        self.widget.applyCalibration(self.calibration)  # pragma: no cover

    def applyAll(self) -> None:
        self.viewspace.applyCalibration(self.calibration)  # pragma: no cover

    def isComplete(self) -> bool:
        return self.table.isComplete()

    def refresh(self) -> None:
        element = self.combo_element.currentText()
        if element not in self.widget.laser.elements:  # pragma: no cover
            return

        data = self.widget.laser.get(element, calibrate=False, flat=True)

        x0, x1, y0, y1 = self.widget.laser.config.data_extent(data.shape)
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

        self.graphics.drawImage(data, rect, self.combo_element.currentText())

        self.graphics.label.setText(self.combo_element.currentText())

        self.graphics.setOverlayItemVisibility()
        self.graphics.updateForeground()
        self.graphics.invalidateScene()

        if len(self.graphics.levels) != self.spinbox_levels.value():
            self.graphics.drawLevels(self.table.ROW_LABELS, self.spinbox_levels.value())

        self.updateCounts()

    def updateWeights(self) -> None:
        element = self.combo_element.currentText()
        wstr = self.combo_weighting.currentText()
        if wstr == "1/σ²":
            if self.calibration[element].x.size > 0:
                shape = self.graphics.data.shape
                levels = self.graphics.currentLevelDataCoords()
                buckets = []
                for (x1, y1, x2, y2) in levels:
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, shape[1]), min(y2, shape[0])
                    bucket = self.graphics.data[y1:y2, x1:x2]
                    buckets.append(bucket)
                weights = 1.0 / np.square(
                    np.array([np.nanstd(bucket) for bucket in buckets])
                )
            else:
                weights = np.empty(0, dtype=np.float64)
            self.calibration[element].weights = (wstr, weights)
        else:
            if wstr == "1/x²":
                wstr = "1/(x^2)"
            elif wstr == "1/y²":
                wstr = "1/(y^2)"
            self.calibration[element].weights = wstr

    def updateCounts(self) -> None:
        if self.graphics.data.size == 0:  # pragma: no cover
            return

        shape = self.graphics.data.shape
        levels = self.graphics.currentLevelDataCoords()
        buckets = []
        for (x1, y1, x2, y2) in levels:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, shape[1]), min(y2, shape[0])
            bucket = self.graphics.data[y1:y2, x1:x2]
            buckets.append(bucket)
        self.table.setCounts(np.array([np.nanmean(bucket) for bucket in buckets]))

    def updateResults(self) -> None:
        # Make sure weights are up to date
        self.updateWeights()
        # Clear results if not complete
        if not self.isComplete():
            self.results.clearResults()
            self.button_plot.setEnabled(False)
        else:
            element = self.combo_element.currentText()

            self.calibration[element].update_linreg()
            self.results.updateResults(self.calibration[element])
            self.button_plot.setEnabled(True)
            if self.dlg is not None:
                self.dlg.updateChart(self.calibration[element])

    # Widget callbacks
    def comboElement(self, text: str) -> None:
        element = self.combo_element.currentText()
        self.table.model().setCalibration(self.calibration[element], resize=False)

        self.lineedit_units.setText(self.calibration[element].unit)

        if self.calibration[element].weighting is not None:
            self.combo_weighting.setCurrentText(self.calibration[element].weighting)
        else:  # pragma: no cover
            self.calibration[element].weighting = "Equal"

        self.refresh()

    def comboWeighting(self, index: int) -> None:
        element = self.combo_element.currentText()
        self.calibration[element].weighting = self.combo_weighting.currentText()
        self.updateResults()

    def lineeditUnits(self) -> None:
        element = self.combo_element.currentText()
        unit = self.lineedit_units.text()
        self.calibration[element].unit = unit

    def showCurve(self) -> QtWidgets.QDialog:
        self.dlg = CalibrationCurveDialog(
            self.combo_element.currentText(),
            self.calibration[self.combo_element.currentText()],
            parent=self,
        )
        self.dlg.finished.connect(self.clearCurve)
        self.dlg.show()
        return self.dlg

    def clearCurve(self) -> None:
        self.dlg = None

    def spinBoxLevels(self) -> None:
        self.table.model().setRowCount(self.spinbox_levels.value())
        self.table.updateGeometry()
        self.refresh()


class StandardsResultsTable(BasicTable):
    LABELS = ["r²", "Gradient", "Intercept", "Sxy", "LOD (3σ)"]

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(2, 5, parent)
        self.setMinimumSize(QtCore.QSize(0, 1))
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setSortingEnabled(False)
        for r in range(2):
            for c in range(len(StandardsResultsTable.LABELS)):
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.setItem(r, c, item)

        for i, label in enumerate(StandardsResultsTable.LABELS):
            self.item(0, i).setText(label)

    def clearResults(self) -> None:
        for i in range(len(StandardsResultsTable.LABELS)):
            self.item(1, i).setText("")

    def updateResults(self, calibration: Calibration) -> None:
        lod = 3.0 * calibration.error / calibration.gradient if calibration.error is not None else np.nan
        for i, v in enumerate(
            [
                calibration.rsq,
                calibration.gradient,
                calibration.intercept,
                calibration.error,
                lod,
            ]
        ):
            item = self.item(1, i)
            item.setText(f"{v:.4f}")


class StandardsTable(BasicTableView):
    ROW_LABELS = [c for c in "ABCDEFGHIJKLMNOPQRST"]
    COLUMN_CONC = 0
    COLUMN_COUNT = 1
    COLUMN_WEIGHTS = 2

    def __init__(self, calibration: Calibration, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        model = CalibrationPointsTableModel(calibration, parent=self)
        self.setModel(model)

        self.hideColumn(2)  # Hide weights column
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self.setItemDelegate(DoubleSignificantFiguresDelegate(4))

    def isComplete(self) -> bool:
        if np.nan in self.model().array[:, StandardsTable.COLUMN_COUNT]:
            return False
        if (
            np.count_nonzero(
                np.nan_to_num(self.model().array[:, StandardsTable.COLUMN_CONC])
            )
            < 2
        ):
            return False
        return True

    def setCounts(self, counts: np.ndarray) -> None:
        self.model().blockSignals(True)
        for i in range(0, self.model().rowCount()):
            self.model().setData(
                self.model().index(i, StandardsTable.COLUMN_COUNT), counts[i]
            )
        self.model().blockSignals(False)
        self.model().dataChanged.emit(
            QtCore.QModelIndex(), QtCore.QModelIndex(), [QtCore.Qt.EditRole]
        )


class CalibrationRectItem(ResizeableRectItem):
    def __init__(
        self,
        rect: QtCore.QRectF,
        label: str,
        px: float,
        py: float,
        font: QtGui.QFont = None,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(rect, parent=parent)

        pen = QtGui.QPen(QtCore.Qt.white, 2.0)
        pen.setCosmetic(True)
        self.setPen(pen)

        if font is None:
            font = QtGui.QFont()

        self.font = font
        self.label = label

        self.px = px
        self.py = py

        self.changed = False

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):
        super().paint(painter, option, widget)

        if self.isSelected():
            painter.fillRect(self.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 32)))

        painter.setFont(self.font)
        fm = painter.fontMetrics()

        pos = QtCore.QPointF(self.rect().left(), self.rect().top())
        pos = painter.transform().map(pos)
        painter.save()
        painter.resetTransform()
        painter.setPen(self.pen())
        painter.drawText(pos.x() + 5, pos.y() + fm.ascent(), self.label)
        painter.restore()

    def itemChange(
        self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value: Any
    ) -> Any:
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            pos = QtCore.QPointF(value)
            pos.setX(pos.x() - pos.x() % self.px)
            pos.setY(pos.y() - pos.y() % self.py)
            self.changed = True
            return pos
        elif change == QtWidgets.QGraphicsItem.ItemSelectedChange:
            if value == 1:
                self.setZValue(self.zValue() + 100)
            else:
                self.setZValue(self.zValue() - 100)
            return value
        return super().itemChange(change, value)

    def selectedSiblings(self) -> List["CalibrationRectItem"]:
        return [
            item
            for item in self.scene().selectedItems()
            if isinstance(item, CalibrationRectItem)
        ]

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        eventpos = self.itemChange(
            QtWidgets.QGraphicsItem.ItemPositionChange, event.position()
        )
        if self.selected_edge is None:
            super().mouseMoveEvent(event)
        else:
            for item in self.selectedSiblings():
                pos = item.mapFromItem(self, eventpos)
                rect = item.rect()
                if self.selected_edge.startswith("top") and pos.y() < rect.bottom():
                    rect.setTop(pos.y())
                elif self.selected_edge.startswith("bottom") and pos.y() > rect.top():
                    rect.setBottom(pos.y())
                if self.selected_edge.endswith("left") and pos.x() < rect.right():
                    rect.setLeft(pos.x())
                elif self.selected_edge.endswith("right") and pos.x() > rect.left():
                    rect.setRight(pos.x())

                item.setRect(rect)
                item.prepareGeometryChange()
