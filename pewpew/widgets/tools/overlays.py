import numpy as np
from pathlib import Path

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib.process.calc import normalise

from pewpew.actions import qAction, qToolButton

from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.items import ScaledImageItem
from pewpew.graphics.overlaygraphics import OverlayScene, OverlayView
from pewpew.graphics.overlayitems import MetricScaleBarOverlay

from pewpew.widgets.exportdialogs import _ExportDialogBase, PngOptionsBox
from pewpew.widgets.ext import RangeSlider
from pewpew.widgets.laser import LaserWidget
from pewpew.widgets.prompts import OverwriteFilePrompt
from pewpew.widgets.tools import ToolWidget

from typing import Iterator, Generator, List, Tuple


class RGBLabelItem(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        texts: List[str],
        colors: List[QtGui.QColor],
        font: QtGui.QFont = None,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(parent)

        if font is None:
            font = QtGui.QFont()

        self._texts = texts
        self.colors = colors
        self.font = font

    @property
    def texts(self) -> List[str]:
        return self._texts

    @texts.setter
    def texts(self, texts: List[str]) -> None:
        self._texts = texts
        self.prepareGeometryChange()

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font)
        width = max((fm.width(text) for text in self._texts), default=0.0)
        return QtCore.QRectF(0, 0, width, fm.height() * len(self._texts))

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):

        fm = QtGui.QFontMetrics(self.font, painter.device())
        y = fm.ascent()
        for text, color in zip(self._texts, self.colors):
            path = QtGui.QPainterPath()
            path.addText(0, y, self.font, text)

            painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
            painter.fillPath(path, QtGui.QBrush(color, QtCore.Qt.SolidPattern))

            y += fm.height()


class RGBOverlayView(OverlayView):
    def __init__(self, options: GraphicsOptions, parent: QtWidgets.QWidget = None):
        self.options = options
        self.data: np.ndarray = None

        self._scene = OverlayScene(0, 0, 640, 480)
        self._scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

        super().__init__(self._scene, parent)
        self.cursors["selection"] = QtCore.Qt.ArrowCursor

        self.image: ScaledImageItem = None

        self.label = RGBLabelItem(
            ["_"], colors=[self.options.font_color], font=self.options.font
        )
        self.scalebar = MetricScaleBarOverlay(
            font=self.options.font, color=self.options.font_color
        )

        self.scene().addOverlayItem(
            self.label,
            QtCore.Qt.TopLeftCorner,
            QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft,
        )
        self.label.setPos(10, 10)
        self.scene().addOverlayItem(
            self.scalebar,
            QtCore.Qt.TopRightCorner,
            QtCore.Qt.AlignTop | QtCore.Qt.AlignRight,
        )
        self.scalebar.setPos(0, 10)

    def setOverlayItemVisibility(
        self, label: bool = None, scalebar: bool = None, colorbar: bool = None
    ):
        if label is None:
            label = self.options.items["label"]
        if scalebar is None:
            scalebar = self.options.items["scalebar"]
        # if colorbar is None:
        #     colorbar = self.options.items["colorbar"]

        self.label.setVisible(label)
        self.scalebar.setVisible(scalebar)
        # self.colorbar.setVisible(colorbar)

    def drawImage(self, data: np.ndarray, rect: QtCore.QRectF) -> None:
        if self.image is not None:
            self.scene().removeItem(self.image)

        self.image = ScaledImageItem.fromArray(
            data, rect, smooth=self.options.smoothing
        )
        self.scene().addItem(self.image)

        if self.sceneRect() != rect:
            self.setSceneRect(rect)
            self.fitInView(rect, QtCore.Qt.KeepAspectRatio)


class OverlayTool(ToolWidget):
    model_type = {"any": "additive", "cmyk": "subtractive", "rgb": "additive"}

    def __init__(self, widget: LaserWidget):
        super().__init__(
            widget, control_label="", orientation=QtCore.Qt.Vertical, apply_all=False
        )
        self.setWindowTitle("Image Overlay")

        self.button_save = QtWidgets.QPushButton("Export")
        self.button_save.setIcon(QtGui.QIcon.fromTheme("document-export"))
        self.button_save.pressed.connect(self.openExportDialog)

        self.graphics = RGBOverlayView(self.viewspace.options)
        # self.graphics.cursorClear.connect(self.widget.clearCursorStatus)
        # self.graphics.cursorMoved.connect(self.updateCursorStatus)

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
        self.combo_add.addItem("Add Isotope")
        self.combo_add.addItems(widget.laser.isotopes)
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

        layout_graphics = QtWidgets.QVBoxLayout()
        layout_graphics.addWidget(self.graphics)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addLayout(layout_row_bar, 0)
        layout_controls.addWidget(self.rows, 1)

        self.box_graphics.setLayout(layout_graphics)
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
        dlg = OverlayExportDialog(self)
        dlg.open()
        return dlg

    def comboAdd(self, index: int) -> None:
        if index == 0:  # pragma: no cover
            return
        text = self.combo_add.itemText(index)
        self.addRow(text)
        self.combo_add.setCurrentIndex(0)

    def addRow(self, label: str) -> None:
        img = self.widget.laser.get(label, calibrate=True, flat=True)
        vmin, vmax = self.viewspace.options.get_colorrange_as_percentile(label, img)
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
        img = self.widget.laser.get(row.label_name.text(), calibrate=True, flat=True)
        vmin, vmax = row.getVmin(img), row.getVmax(img)

        r, g, b, _ = row.getColor().getRgb()
        if self.model_type[self.rows.color_model] == "subtractive":
            r, g, b = 255 - r, 255 - g, 255 - b

        # Normalise to range
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin)
        # Convert to separate rgb channels
        img = (img[:, :, None] * np.array([r, g, b])).astype(np.uint8)

        return img

    def refresh(self) -> None:
        rows = [row for row in self.rows if not row.hidden]
        datas = [self.processRow(row) for row in rows]

        if len(datas) == 0:
            img = np.zeros((*self.widget.laser.shape[:2], 3), dtype=np.uint32)
        else:
            img = np.sum(datas, axis=0).astype(np.uint32)

        if self.model_type[self.rows.color_model] == "subtractive":
            img = np.full_like(img, 255) - img

        if self.check_normalise.isChecked() and self.check_normalise.isEnabled():
            img = normalise(img, 0, 255).astype(np.uint32)
        else:
            img = np.clip(img, 0, 255)

        img = (255 << 24) + (img[:, :, 0] << 16) + (img[:, :, 1] << 8) + img[:, :, 2]

        x0, x1, y0, y1 = self.widget.laser.config.data_extent(img.shape)
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

        self.graphics.drawImage(img, rect)

        self.graphics.label.colors = [row.getColor() for row in rows]
        self.graphics.label.texts = [row.label_name.text() for row in rows]

        self.graphics.setOverlayItemVisibility()
        self.graphics.updateForeground()
        self.graphics.invalidateScene()

    def updateCursorStatus(self, v: np.ndarray) -> None:
        status_bar = self.viewspace.window().statusBar()
        if status_bar is None:
            return
        status_bar.showMessage(f"r: {v[0]:.2f}, g: {v[1]:.2f}, b: {v[2]:.2f}")


class OverlayItemRow(QtWidgets.QWidget):
    closeRequested = QtCore.Signal("QWidget*")
    itemChanged = QtCore.Signal()

    def __init__(
        self,
        label: str,
        vmin: int,
        vmax: int,
        color: QtGui.QColor,
        color_pickable: bool = False,
        parent: "OverlayRows" = None,
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
            "Select the color for this isotope.",
            self.selectColor,
        )
        self.action_close = qAction(
            "window-close", "Remove", "Remove this isotope.", self.close
        )
        self.action_hide = qAction(
            "visibility", "Hide", "Toggle visibility of this isotope.", self.hideChanged
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

    def __init__(self, parent: QtWidgets.QWidget = None):
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


# Todo: this export isnt real!
class OverlayExportDialog(_ExportDialogBase):
    def __init__(self, parent: OverlayTool):
        super().__init__([PngOptionsBox()], parent)
        self.setWindowTitle("Overlay Export")
        self.widget = parent

        self.check_individual = QtWidgets.QCheckBox("Export colors individually.")
        self.check_individual.setToolTip(
            "Export each color layer as a separte file.\n"
            "The filename will be appended with the colors name."
        )
        self.check_individual.stateChanged.connect(self.updatePreview)

        self.layout.insertWidget(2, self.check_individual)

        path = (
            self.widget.widget.laser.path.with_name(self.widget.widget.laser.name)
            .with_suffix(".png")
            .resolve()
        )
        self.lineedit_directory.setText(str(path.parent))
        self.lineedit_filename.setText(str(path.name))
        self.typeChanged(0)

    def isIndividual(self) -> bool:
        return self.check_individual.isChecked() and self.check_individual.isEnabled()

    def updatePreview(self) -> None:
        path = Path(self.lineedit_filename.text())
        if self.isIndividual():
            if self.widget.rows.color_model == "rgb":
                path = path.with_name(path.stem + "_<rgb>" + path.suffix)
            elif self.widget.rows.color_model == "cmyk":
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
        color_model = self.widget.rows.color_model
        if color_model == "rgb":
            r, g, b, _a = self.widget.rows[row].getColor().getRgb()
            suffix = "r" if r > 0 else "g" if g > 0 else "b"
        elif color_model == "cmyk":
            c, m, y, _k, _a = self.widget.rows[row].getColor().getCmyk()
            suffix = "c" if c > 0 else "m" if m > 0 else "y"
        else:
            suffix = str(row)

        path = self.getPath()
        return path.with_name(path.stem + "_" + suffix + path.suffix)

    def generateRowPaths(self) -> Generator[Tuple[Path, int], None, None]:
        for i, row in enumerate(self.widget.rows):
            if not row.hidden:
                yield (self.getPathForRow(i), i)

    def export(self, path: Path) -> None:
        option = self.options.currentOption()

        if option.ext == ".png":
            if option.raw():
                self.widget.graphics.image.image.save(str(path.absolute()))
            else:
                self.widget.graphics.saveToFile(str(path.absolute()))
        else:
            raise ValueError(f"Unable to export file as '{option.ext}'.")

    def exportIndividual(self, path: Path, rowi: int) -> None:
        option = self.options.currentOption()
        row = self.widget.rows[rowi]

        if option.ext == ".png":
            img = self.widget.processRow(row).astype(np.uint32)

            if OverlayTool.model_type[self.widget.rows.color_model] == "subtractive":
                img = np.full_like(img, 255) - img

            img = (
                (255 << 24) + (img[:, :, 0] << 16) + (img[:, :, 1] << 8) + img[:, :, 2]
            )

            x0, x1, y0, y1 = self.widget.widget.laser.config.data_extent(img.shape)
            rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

            if option.raw():
                image = QtGui.QImage(
                    img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB32
                )
                image.save(str(path.absolute()))
            else:
                self.widget.graphics.drawImage(img, rect)

                self.widget.graphics.label.colors = [QtCore.Qt.white]
                self.widget.graphics.label.texts = [row.label_name.text()]

                self.widget.graphics.updateForeground()
                self.widget.graphics.invalidateScene()

                pixmap = QtGui.QPixmap(self.widget.graphics.viewport().size())

                painter = QtGui.QPainter(pixmap)
                self.widget.graphics.render(painter)
                painter.end()

                pixmap.save(str(path.absolute()))
        else:
            raise ValueError(f"Unable to export file as '{option.ext}'.")

    def accept(self) -> None:
        prompt = OverwriteFilePrompt()
        if self.isIndividual():
            paths = [p for p in self.generateRowPaths() if prompt.promptOverwrite(p[0])]
            if len(paths) == 0:
                return
        else:
            if not prompt.promptOverwrite(self.getPath()):
                return

        if self.isIndividual():
            for path, row in paths:
                self.exportIndividual(path, row)
            self.widget.refresh()
        else:
            self.export(self.getPath())

        super().accept()
