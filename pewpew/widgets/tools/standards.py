import copy
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

from pewlib import Calibration
from pewpew.graphics import colortable
from pewpew.graphics.imageitems import LaserImageItem, ScaledImageItem, SnapImageItem

from pewpew.graphics.items import ResizeableRectItem

from pewpew.validators import DoubleSignificantFiguresDelegate

from pewpew.widgets.dialogs import CalibrationCurveDialog
from pewpew.models import CalibrationPointsTableModel
from pewpew.widgets.modelviews import (
    BasicTable,
    BasicTableView,
)
from pewpew.widgets.views import TabView

from .tool import ToolWidget

from typing import Any, Dict, List, Optional


class StandardsTool(ToolWidget):
    """Tool for creating calibrations from a laser image."""

    WEIGHTINGS = ["Equal", "1/σ²", "x", "1/x", "1/x²", "y", "1/y", "1/y²"]

    def __init__(self, item: LaserImageItem, view: Optional[TabView]):
        super().__init__(item, control_label="Calibration", apply_all=True, view=view)
        self.setWindowTitle("Calibration Tool")

        self.graphics.setInteractionFlag("tool")
        self.graphics.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.image: Optional[ScaledImageItem] = None
        self.levels: List[CalibrationRectItem] = []

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
        # self.graphics = StandardsGraphicsView(self.viewspace.options, parent=self)
        # self.levelsChanged.connect(self.updateCounts)

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.currentIndexChanged.connect(self.comboElement)

        self.table = StandardsTable(Calibration(), self)
        self.table.model().setRowCount(6)
        self.table.model().dataChanged.connect(self.completeChanged)
        self.table.model().dataChanged.connect(self.updateResults)

        # Initialise
        self.calibration = copy.deepcopy(self.item.laser.calibration)
        # Prevent currentIndexChanged being emmited
        self.combo_element.blockSignals(True)
        self.combo_element.clear()
        self.combo_element.addItems(self.item.laser.elements)
        self.combo_element.setCurrentText(self.item.element())
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

        self.box_graphics.layout().addLayout(layout_results, 0)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addLayout(layout_cal_form)
        layout_controls.addWidget(self.table)
        layout_controls.addLayout(layout_table_form)
        self.box_controls.setLayout(layout_controls)

        self.refresh()
        self.updateResults()

    def apply(self) -> None:
        self.item.applyCalibration(self.calibration)  # pragma: no cover

    def applyAll(self) -> None:
        self.view.applyCalibration(self.calibration)  # pragma: no cover

    def drawLevels(self, labels: str, n: int) -> None:
        for item in self.levels:
            self.graphics.scene().removeItem(item)
        self.levels = []

        x1 = 0.1 * self.image.boundingRect().width()
        x2 = 0.9 * self.image.boundingRect().width()

        height_per_level = self.image.boundingRect().height() / n

        for i in range(n):
            y1 = self.image.boundingRect().y() + i * height_per_level
            y2 = self.image.boundingRect().y() + (i + 1) * height_per_level

            rect = QtCore.QRectF(
                self.image.snapPos(QtCore.QPointF(x1, y1)),
                self.image.snapPos(QtCore.QPointF(x2, y2)),
            )
            rect.moveTo(0, 0)

            item = CalibrationRectItem(
                rect, labels[i], self.image, font=self.item.options.font
            )
            item.setPos(QtCore.QPointF(x1, y1))
            item.changed.connect(self.updateCounts)
            item.setZValue(self.image.zValue() + 1)
            self.levels.append(item)

    def isComplete(self) -> bool:
        return self.table.isComplete()

    def refresh(self) -> None:
        element = self.combo_element.currentText()
        if element not in self.item.laser.elements:  # pragma: no cover
            return

        data = self.item.laser.get(element, calibrate=False, flat=True)

        x0, x1, y0, y1 = self.item.laser.config.data_extent(data.shape)
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

        vmin, vmax = self.item.options.get_color_range_as_float("<calc>", data)
        data = np.clip(data, vmin, vmax)
        if vmin != vmax:
            data = (data - vmin) / (vmax - vmin)

        table = colortable.get_table(self.item.options.colortable)

        image = ScaledImageItem.fromArray(data, rect, table)
        self.graphics.scene().addItem(image)
        for item in self.levels:
            item.setParentItem(image)
        if self.image is not None:
            self.graphics.scene().removeItem(self.image)
        self.image = image

        self.colorbar.updateTable(
            table, vmin, vmax, self.item.laser.calibration[element].unit
        )

        if len(self.levels) != self.spinbox_levels.value():
            self.drawLevels(self.table.ROW_LABELS, self.spinbox_levels.value())

        self.graphics.invalidateScene()

        self.updateCounts()

    def updateWeights(self) -> None:
        element = self.combo_element.currentText()
        wstr = self.combo_weighting.currentText()
        if wstr == "1/σ²":
            if self.calibration[element].x.size > 0:
                buckets = []
                for item in self.levels:
                    p1 = self.image.mapToData(item.mapToScene(item.rect.topLeft()))
                    p2 = self.image.mapToData(item.mapToScene(item.rect.bottomRight()))
                    p1.setX(max(p1.x(), 0))
                    p2.setX(max(p2.x(), 0))
                    p1.setY(max(p1.y(), 0))
                    p2.setY(max(p2.y(), 0))
                    bucket = self.item.laser.data[element][
                        p1.y() : p2.y(), p1.x() : p2.x()
                    ]
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
        element = self.combo_element.currentText()
        if self.image is None:
            return

        buckets = []
        for item in self.levels:
            p1 = self.image.mapToData(item.mapToScene(item.rect.topLeft()))
            p2 = self.image.mapToData(item.mapToScene(item.rect.bottomRight()))
            p1.setX(max(p1.x(), 0))
            p2.setX(max(p2.x(), 0))
            p1.setY(max(p1.y(), 0))
            p2.setY(max(p2.y(), 0))
            bucket = self.item.laser.data[element][p1.y() : p2.y(), p1.x() : p2.x()]
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

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
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
        lod = (
            3.0 * calibration.error / calibration.gradient
            if calibration.error is not None
            else np.nan
        )
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

    def __init__(
        self, calibration: Calibration, parent: Optional[QtWidgets.QWidget] = None
    ):
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
                ~np.isnan(self.model().array[:, StandardsTable.COLUMN_CONC])
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
    changed = QtCore.Signal()

    def __init__(
        self,
        rect: QtCore.QRectF,
        label: str,
        item: SnapImageItem,
        font: Optional[QtGui.QFont] = None,
    ):
        super().__init__(rect, parent=item)
        self.item = item

        pen = QtGui.QPen(QtCore.Qt.white, 2.0)
        pen.setCosmetic(True)
        self.setPen(pen)

        if font is None:
            font = QtGui.QFont()

        self.font = font
        self.label = label

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        super().paint(painter, option, widget)

        if self.isSelected():
            painter.fillRect(self.rect, QtGui.QBrush(QtGui.QColor(255, 255, 255, 32)))

        painter.setFont(self.font)
        fm = painter.fontMetrics()

        pos = QtCore.QPointF(self.rect.left(), self.rect.top())
        pos = painter.transform().map(pos)
        painter.save()
        painter.resetTransform()
        painter.setPen(self.pen)
        painter.drawText(pos.x() + 5, pos.y() + fm.ascent(), self.label)
        painter.restore()

    def itemChange(
        self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value: Any
    ) -> Any:
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            pos = QtCore.QPointF(value)
            pos = self.item.snapPos(pos)
            if self.pos() != pos:
                self.changed.emit()
            return pos
        elif change == QtWidgets.QGraphicsItem.ItemSelectedChange:
            if value == 1:
                self.setZValue(self.zValue() + 1)
            else:
                self.setZValue(self.zValue() - 1)
            return value
        return super().itemChange(change, value)

    def selectedSiblings(self) -> List["CalibrationRectItem"]:
        return [
            item
            for item in self.parentItem().childItems()
            if isinstance(item, CalibrationRectItem) and item.isSelected()
        ]


    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.selected_edge is None:
            super().mouseMoveEvent(event)
        else:
            for item in self.selectedSiblings():
                pos = item.mapFromItem(self, self.item.snapPos(event.pos()))
                if (
                    self.selected_edge.startswith("top")
                    and pos.y() < item.rect.bottom()
                ):
                    item.rect.setTop(pos.y())
                elif (
                    self.selected_edge.startswith("bottom")
                    and pos.y() > item.rect.top()
                ):
                    item.rect.setBottom(pos.y())
                if self.selected_edge.endswith("left") and pos.x() < item.rect.right():
                    item.rect.setLeft(pos.x())
                elif (
                    self.selected_edge.endswith("right") and pos.x() > item.rect.left()
                ):
                    item.rect.setRight(pos.x())

                item.prepareGeometryChange()
            self.changed.emit()
