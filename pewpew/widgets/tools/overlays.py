import numpy as np
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from pewlib.process.calc import normalise

from pewpew.actions import qAction, qToolButton

from pewpew.graphics.overlayitems import OverlayItem
from pewpew.graphics.imageitems import LaserImageItem, ScaledImageItem
from pewpew.lib.numpyqt import array_to_image

from pewpew.widgets.exportdialogs import _ExportDialogBase, OptionsBox
from pewpew.widgets.ext import RangeSlider
from pewpew.widgets.prompts import OverwriteFilePrompt
from pewpew.widgets.tools import ToolWidget
from pewpew.widgets.views import TabView

from typing import Iterator, Generator, List,  Tuple


class OverlayLabelItem(OverlayItem):
    def __init__(
        self,
        font: QtGui.QFont | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(parent)

        self.font = font or QtGui.QFont()
        self.texts = []
        self.colors = []

    def setColors(self, colors: List[QtGui.QColor]) -> None:
        self.colors = colors

    def setTexts(self, texts: List[str]) -> None:
        self.texts = texts
        self.prepareGeometryChange()

    def boundingRect(self) -> QtCore.QRectF:
        fm = QtGui.QFontMetrics(self.font)
        return QtCore.QRectF(
            0,
            0,
            max([fm.boundingRect(text).width() for text in self.texts] + [0]),
            fm.height() * len(self.texts),
        )

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        fm = QtGui.QFontMetrics(self.font, painter.device())
        y = fm.ascent()
        for text, color in zip(self.texts, self.colors):
            path = QtGui.QPainterPath()
            path.addText(0, y, self.font, text)

            painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
            painter.fillPath(path, QtGui.QBrush(color, QtCore.Qt.SolidPattern))

            y += fm.height()

        painter.restore()


class OverlayTool(ToolWidget):
    """Tool for displaying and exporting color overlays."""

    model_type = {"any": "additive", "cmyk": "subtractive", "rgb": "additive"}

    def __init__(self, item: LaserImageItem, view: TabView | None = None):
        super().__init__(
            item,
            control_label="",
            orientation=QtCore.Qt.Vertical,
            apply_all=False,
            view=view,
        )
        self.setWindowTitle("Image Overlay")

        self.image: ScaledImageItem | None = None

        self.label = OverlayLabelItem(self.item.options.font)
        self.label.setPos(10, 10)
        self.graphics.addOverlayItem(self.label)

        self.colorbar.setVisible(False)

        self.button_save = QtWidgets.QPushButton("Export")
        self.button_save.setIcon(QtGui.QIcon.fromTheme("document-export"))
        self.button_save.pressed.connect(self.openExportDialog)

        self.check_normalise = QtWidgets.QCheckBox("Renormalise")
        self.check_normalise.setEnabled(False)
        self.check_normalise.clicked.connect(self.refresh)

        self.radio_rgb = QtWidgets.QRadioButton("rgb")
        self.radio_rgb.setChecked(True)
        self.radio_rgb.toggled.connect(self.updateColorModel)
        self.radio_cmyk = QtWidgets.QRadioButton("cmyk")
        self.radio_cmyk.toggled.connect(self.updateColorModel)
        self.radio_custom = QtWidgets.QRadioButton("any")
        self.radio_custom.toggled.connect(self.updateColorModel)

        self.combo_add = QtWidgets.QComboBox()
        self.combo_add.addItem("Add Element")
        self.combo_add.addItems(item.laser.elements)
        self.combo_add.activated[int].connect(self.comboAdd)

        self.rows = OverlayRows(self)
        self.rows.rowsChanged.connect(self.rowsChanged)
        self.rows.rowsChanged.connect(self.completeChanged)
        self.rows.itemChanged.connect(self.refresh)

        layout_row_bar = QtWidgets.QHBoxLayout()
        layout_row_bar.addWidget(self.combo_add, 1, QtCore.Qt.AlignLeft)
        layout_row_bar.addWidget(self.check_normalise, 0, QtCore.Qt.AlignRight)
        layout_row_bar.addWidget(self.radio_rgb, 0, QtCore.Qt.AlignRight)
        layout_row_bar.addWidget(self.radio_cmyk, 0, QtCore.Qt.AlignRight)
        layout_row_bar.addWidget(self.radio_custom, 0, QtCore.Qt.AlignRight)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addLayout(layout_row_bar, 0)
        layout_controls.addWidget(self.rows, 1)

        self.box_controls.setLayout(layout_controls)

        self.button_box.clear()
        self.button_box.addButton(self.button_save, QtWidgets.QDialogButtonBox.YesRole)
        self.button_box.addButton(QtWidgets.QDialogButtonBox.Cancel)

        self.refresh()

    def isComplete(self) -> bool:
        return self.rows.rowCount() > 0

    @QtCore.Slot()
    def completeChanged(self) -> None:
        enabled = self.isComplete()
        self.button_save.setEnabled(enabled)

    def openExportDialog(self) -> QtWidgets.QDialog:
        if self.image is None:
            raise ValueError

        path = Path(self.item.laser.info.get("File Path", ""))
        path = path.with_name(self.item.laser.info.get("Name", "laser") + ".png")

        dlg = OverlayExportDialog(
            path,
            self.image,
            [self.processRow(row) if not row.hidden else None for row in self.rows],
            self.rows.color_model,
            self,
        )
        dlg.open()
        return dlg

    def comboAdd(self, index: int) -> None:
        if index == 0:  # pragma: no cover
            return
        text = self.combo_add.itemText(index)
        self.addRow(text)
        self.combo_add.setCurrentIndex(0)

    def addRow(self, label: str) -> None:
        img = self.item.laser.get(label, calibrate=True, flat=True)
        vmin, vmax = self.item.options.get_color_range_as_percentile(label, img)
        self.rows.addRow(label, int(vmin), int(vmax))

    def updateColorModel(self) -> None:
        color_model = (
            "rgb"
            if self.radio_rgb.isChecked()
            else "cmyk"
            if self.radio_cmyk.isChecked()
            else "any"
        )
        self.rows.setColorModel(color_model)
        self.rowsChanged(self.rows.rowCount())

    def rowsChanged(self, rows: int) -> None:
        if self.rows.color_model in ["rgb", "cmyk"]:
            self.combo_add.setEnabled(rows < 3)
            self.check_normalise.setEnabled(False)
        else:
            self.combo_add.setEnabled(True)
            self.check_normalise.setEnabled(True)
        self.refresh()

    def processRow(self, row: "OverlayItemRow") -> np.ndarray:
        img = self.item.laser.get(row.label_name.text(), calibrate=True, flat=True)
        vmin, vmax = row.getVmin(img), row.getVmax(img)

        r, g, b, _ = row.getColor().getRgb()
        if self.model_type[self.rows.color_model] == "subtractive":
            r, g, b = 255 - r, 255 - g, 255 - b

        # Normalise to range
        img = np.clip(img, vmin, vmax)
        if vmin != vmax:
            img = (img - vmin) / (vmax - vmin)
        # Convert to separate rgb channels
        img = (img[:, :, None] * np.array([r, g, b])).astype(np.uint8)

        return img

    def refresh(self) -> None:
        rows = [row for row in self.rows if not row.hidden]
        datas = np.array([self.processRow(row) for row in rows])

        if len(datas) == 0:
            img = np.zeros((*self.item.laser.shape[:2], 3), dtype=np.uint32)
        else:
            img = np.sum(datas, axis=0).astype(np.uint32)

        if self.model_type[self.rows.color_model] == "subtractive":
            img = np.full_like(img, 255) - img

        if self.check_normalise.isChecked() and self.check_normalise.isEnabled():
            img = normalise(img, 0, 255).astype(np.uint32)
        else:
            img = np.clip(img, 0, 255)

        img = (255 << 24) + (img[:, :, 0] << 16) + (img[:, :, 1] << 8) + img[:, :, 2]

        x0, x1, y0, y1 = self.item.laser.config.data_extent(img.shape)
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

        if self.image is not None:
            self.graphics.scene().removeItem(self.image)

        self.image = ScaledImageItem.fromArray(img, rect)
        self.graphics.scene().addItem(self.image)

        self.label.setColors([row.getColor() for row in rows])
        self.label.setTexts([row.label_name.text() for row in rows])

        self.graphics.invalidateScene()

    def updateCursorStatus(self, v: np.ndarray) -> None:
        status_bar = self.viewspace.window().statusBar()
        if status_bar is None:
            return
        status_bar.showMessage(f"r: {v[0]:.2f}, g: {v[1]:.2f}, b: {v[2]:.2f}")


class OverlayItemRow(QtWidgets.QWidget):
    closeRequested = QtCore.Signal(QtWidgets.QWidget)
    itemChanged = QtCore.Signal()

    def __init__(
        self,
        label: str,
        vmin: int,
        vmax: int,
        color: QtGui.QColor,
        color_pickable: bool = False,
        parent: "OverlayRows" | None = None,
    ):
        super().__init__(parent)

        self.label_name = QtWidgets.QLabel(label)
        self.colorrange = RangeSlider()
        self.colorrange.setRange(0, 99)
        self.colorrange.setValues(vmin, vmax)
        self.colorrange.valueChanged.connect(self.itemChanged)
        self.colorrange.value2Changed.connect(self.itemChanged)

        self.action_color = qAction(
            "color-picker",
            "Color",
            "Select the color for this element.",
            self.selectColor,
        )
        self.action_close = qAction(
            "window-close", "Remove", "Remove this element.", self.close
        )
        self.action_hide = qAction(
            "visibility", "Hide", "Toggle visibility of this element.", self.hideChanged
        )
        self.action_hide.setCheckable(True)

        self.button_hide = qToolButton(action=self.action_hide)
        self.button_color = qToolButton(action=self.action_color)
        self.button_color.setEnabled(color_pickable)

        self.effect_color = QtWidgets.QGraphicsColorizeEffect()
        self.effect_color.setColor(color)
        self.button_color.setGraphicsEffect(self.effect_color)

        self.button_close = qToolButton(action=self.action_close)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label_name, 2)
        layout.addWidget(self.colorrange, 1)
        layout.addWidget(self.button_hide, 0)
        layout.addWidget(self.button_color, 0)
        layout.addWidget(self.button_close, 0, QtCore.Qt.AlignRight)
        self.setLayout(layout)

    @property
    def hidden(self) -> bool:
        return self.action_hide.isChecked()

    def hideChanged(self) -> None:
        if self.hidden:
            self.button_hide.setIcon(QtGui.QIcon.fromTheme("hint"))
        else:
            self.button_hide.setIcon(QtGui.QIcon.fromTheme("visibility"))
        self.itemChanged.emit()

    def getVmin(self, data: np.ndarray) -> float:
        return np.nanpercentile(data, self.colorrange.left())

    def getVmax(self, data: np.ndarray) -> float:
        return np.nanpercentile(data, self.colorrange.right())

    def getColor(self) -> QtGui.QColor:
        return self.effect_color.color()

    def setColor(self, color: QtGui.QColor) -> None:
        self.effect_color.setColor(color)
        self.itemChanged.emit()

    def selectColor(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QColorDialog(self.getColor(), self)
        dlg.colorSelected.connect(self.setColor)
        dlg.open()
        return dlg

    def setColorPickable(self, pickable: bool = True) -> None:
        self.button_color.setEnabled(pickable)

    def close(self) -> None:
        self.closeRequested.emit(self)
        super().close()


class OverlayRows(QtWidgets.QScrollArea):
    rowsChanged = QtCore.Signal(int)
    itemChanged = QtCore.Signal()

    colors_rgb = [
        QtGui.QColor.fromRgbF(1.0, 0.0, 0.0),
        QtGui.QColor.fromRgbF(0.0, 1.0, 0.0),
        QtGui.QColor.fromRgbF(0.0, 0.0, 1.0),
    ]
    colors_cmyk = [
        QtGui.QColor.fromCmykF(1.0, 0.0, 0.0, 0.0),
        QtGui.QColor.fromCmykF(0.0, 1.0, 0.0, 0.0),
        QtGui.QColor.fromCmykF(0.0, 0.0, 1.0, 0.0),
    ]

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.rows: List[OverlayItemRow] = []
        self.color_model = "rgb"
        self.max_rows = 3

        widget = QtWidgets.QWidget()
        self.setWidget(widget)
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        widget.setLayout(self.layout)

    def __getitem__(self, i: int) -> OverlayItemRow:
        return self.rows[i]

    def __iter__(self) -> Iterator:
        return iter(self.rows)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(500, 34 * 3 + 10)

    def rowCount(self) -> int:
        return len(self.rows)

    def addRow(
        self,
        label: str,
        vmin: int,
        vmax: int,
    ) -> None:
        if self.rowCount() >= self.max_rows:
            return

        pickable = self.color_model == "any"

        row = OverlayItemRow(
            label, vmin, vmax, QtGui.QColor.fromRgbF(0.0, 0.0, 0.0), pickable, self
        )
        row.closeRequested.connect(self.removeRow)
        row.itemChanged.connect(self.itemChanged)

        self.rows.append(row)
        self.layout.addWidget(row)
        self.rowsChanged.emit(self.rowCount())
        self.recolor()

    def removeRow(self, row: OverlayItemRow) -> None:
        self.rows.remove(row)
        self.rowsChanged.emit(self.rowCount())
        self.recolor()

    def recolor(self) -> None:
        if self.color_model == "rgb":
            for i, row in enumerate(self.rows[:3]):
                row.setColor(self.colors_rgb[i])
        elif self.color_model == "cmyk":
            for i, row in enumerate(self.rows[:3]):
                row.setColor(self.colors_cmyk[i])

    def setColorModel(self, color_model: str) -> None:
        if self.color_model == color_model:
            return
        self.color_model = color_model
        if color_model in ["rgb", "cmyk"]:
            self.max_rows = 3
            for row in self.rows[:3]:
                row.setColorPickable(False)
            for row in self.rows[3:]:
                row.close()
            self.recolor()
        else:
            self.max_rows = 99
            for row in self.rows:
                row.setColorPickable(True)


class OverlayExportDialog(_ExportDialogBase):
    """Export dialog for the overlay tool."""

    def __init__(
        self,
        default_path: Path,
        image: ScaledImageItem,
        row_images: List[np.ndarray],
        color_model: str,
        parent: OverlayTool,
    ):
        super().__init__([OptionsBox("PNG images", ".png")], parent)
        self.setWindowTitle("Overlay Export")

        self.image = image
        self.row_images = row_images
        self.color_model = color_model

        self.check_individual = QtWidgets.QCheckBox("Export colors individually.")
        self.check_individual.setToolTip(
            "Export each color layer as a separte file.\n"
            "The filename will be appended with the colors name."
        )
        self.check_individual.stateChanged.connect(self.updatePreview)

        self.layout.insertWidget(2, self.check_individual)

        self.lineedit_directory.setText(str(default_path.parent))
        self.lineedit_filename.setText(str(default_path.name))
        self.typeChanged(0)

    def isIndividual(self) -> bool:
        return self.check_individual.isChecked() and self.check_individual.isEnabled()

    def updatePreview(self) -> None:
        path = Path(self.lineedit_filename.text())
        if self.isIndividual():
            if self.color_model == "rgb":
                path = path.with_name(path.stem + "_<rgb>" + path.suffix)
            elif self.color_model == "cmyk":
                path = path.with_name(path.stem + "_<cmyk>" + path.suffix)
            else:
                path = path.with_name(path.stem + "_<#>" + path.suffix)
        self.lineedit_preview.setText(str(path))

    def selectDirectory(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(self, "Select Directory", "")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.lineedit_directory.setText)
        dlg.open()
        return dlg

    def getPath(self) -> Path:
        return Path(self.lineedit_directory.text()).joinpath(
            self.lineedit_filename.text()
        )

    def getPathForRow(self, row: int) -> Path:
        if self.color_model == "rgb":
            suffix = "rgb"[row]
        elif self.color_model == "cmyk":
            suffix = "cmyk"[row]
        else:
            suffix = str(row)

        path = self.getPath()
        return path.with_name(path.stem + "_" + suffix + path.suffix)

    def generateRowPaths(self) -> Generator[Tuple[Path, int], None, None]:
        for i, row in enumerate(self.row_images):
            if row is not None:
                yield (self.getPathForRow(i), i)

    def export(self, path: Path) -> None:
        option = self.options.currentOption()

        if option.ext == ".png":
            self.image.image.save(str(path.absolute()))
        else:
            raise ValueError(f"Unable to export file as '{option.ext}'.")

    def exportIndividual(self, path: Path, row: int) -> None:
        option = self.options.currentOption()

        if option.ext == ".png":
            image = array_to_image(self.row_images[row])
            image.save(str(path.absolute()))
        else:
            raise ValueError(f"Unable to export file as '{option.ext}'.")

    def accept(self) -> None:
        prompt = OverwriteFilePrompt()
        if self.isIndividual():
            paths = [p for p in self.generateRowPaths() if prompt.promptOverwrite(p[0])]
            if len(paths) == 0:
                return
            for path, row in paths:
                self.exportIndividual(path, row)
        else:
            if not prompt.promptOverwrite(self.getPath()):
                return
            self.export(self.getPath())

        super().accept()
