import os.path
import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.backend_bases import MouseEvent
from matplotlib.image import imsave
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patheffects import withStroke

from pew import io
from pew.lib.calc import greyscale_to_rgb, normalise

from pewpew.actions import qAction, qToolButton
from pewpew.validators import PercentOrDecimalValidator

from pewpew.lib.mpltools import MetricSizeBar
from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import InteractiveImageCanvas
from pewpew.widgets.exportdialogs import _ExportDialogBase, PngOptionsBox
from pewpew.widgets.laser import LaserWidget
from pewpew.widgets.prompts import OverwriteFilePrompt
from pewpew.widgets.tools import ToolWidget

from typing import Iterator, Generator, List, Tuple, Union


class OverlayTool(ToolWidget):
    model_type = {"any": "additive", "cmyk": "subtractive", "rgb": "additive"}

    def __init__(self, widget: LaserWidget):
        super().__init__(widget, apply_all=False)
        self.setWindowTitle("Image Overlay Tool")

        self.button_box.clear()

        self.button_save = QtWidgets.QPushButton("Export")
        self.button_save.setIcon(QtGui.QIcon.fromTheme("document-export"))
        self.button_save.pressed.connect(self.openExportDialog)

        self.canvas = OverlayCanvas(self.viewspace.options)
        self.canvas.cursorClear.connect(self.widget.clearCursorStatus)
        self.canvas.cursorMoved.connect(self.updateCursorStatus)
        self.canvas.view_limits = self.widget.canvas.view_limits

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

        layout_top = QtWidgets.QHBoxLayout()
        layout_top.addWidget(self.combo_add, 1, QtCore.Qt.AlignLeft)
        layout_top.addWidget(self.radio_rgb, 0, QtCore.Qt.AlignRight)
        layout_top.addWidget(self.radio_cmyk, 0, QtCore.Qt.AlignRight)
        layout_top.addWidget(self.radio_custom, 0, QtCore.Qt.AlignRight)

        self.layout_main.addWidget(self.canvas, 1)
        self.layout_main.addLayout(layout_top)
        self.layout_main.addWidget(self.rows, 0)
        self.layout_main.addWidget(self.check_normalise, 0)

        self.button_box.addButton(self.button_save, QtWidgets.QDialogButtonBox.YesRole)
        self.button_box.addButton(QtWidgets.QDialogButtonBox.Cancel)

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
        vmin, vmax = self.viewspace.options.colors.get_range(label)
        self.rows.addRow(label, vmin, vmax)

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

        r, g, b, _a = row.getColor().getRgbF()
        if self.model_type[self.rows.color_model] == "subtractive":
            r, g, b = 1.0 - r, 1.0 - g, 1.0 - b

        return greyscale_to_rgb(
            (np.clip(img, vmin, vmax) - vmin) / (vmax - vmin), (r, g, b)
        )

    def refresh(self) -> None:
        datas = [self.processRow(row) for row in self.rows if not row.hidden]
        if len(datas) == 0:
            img = np.zeros((*self.widget.laser.shape[:2], 3), float)
        else:
            img = np.sum(datas, axis=0)

        if self.model_type[self.rows.color_model] == "subtractive":
            img = np.ones_like(img) - img

        if self.check_normalise.isChecked() and self.check_normalise.isEnabled():
            img = normalise(img, 0.0, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)

        extent = self.widget.laser.config.data_extent(img.shape)
        # Only change the view if new or the laser extent has changed (i.e. conf edit)
        if self.canvas.extent != extent:
            self.canvas.view_limits = self.canvas.extentForAspect(extent)

        self.canvas.drawData(img, extent)

        if self.viewspace.options.canvas.label:
            names = [row.label_name.text() for row in self.rows if not row.hidden]
            colors = [row.getColor().name() for row in self.rows if not row.hidden]
            self.canvas.drawLabel(names, colors)
        elif self.canvas.label is not None:  # pragma: no cover
            self.canvas.label.remove()
            self.canvas.label = None

        if self.viewspace.options.canvas.scalebar:
            self.canvas.drawScalebar()
        elif self.canvas.scalebar is not None:  # pragma: no cover
            self.canvas.scalebar.remove()
            self.canvas.scalebar = None

        self.canvas.draw_idle()

    def saveCanvas(self, path: str, raw: bool = False) -> None:
        if raw:
            imsave(path, self.canvas.image.get_array())
        else:
            self.canvas.figure.savefig(
                path, dpi=300, bbox_inches="tight", transparent=True, facecolor=None
            )

    def updateCursorStatus(self, v: np.ndarray) -> None:
        status_bar = self.viewspace.window().statusBar()
        if status_bar is None:
            return
        status_bar.showMessage(f"r: {v[0]:.2f}, g: {v[1]:.2f}, b: {v[2]:.2f}")


class OverlayCanvas(InteractiveImageCanvas):
    cursorMoved = QtCore.Signal(np.ndarray)

    def __init__(self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None):
        super().__init__(move_button=1, parent=parent)
        self.viewoptions = viewoptions

        self.label: AnchoredOffsetbox = None
        self.scalebar: MetricSizeBar = None

        self.drawFigure()

    def moveCursor(self, event: MouseEvent) -> None:
        if self.image is not None:
            v = self.image.get_cursor_data(event)
            self.cursorMoved.emit(v)

    def drawLabel(self, names: List[str], colors: List[str]) -> None:
        if self.label is not None:
            self.label.remove()

        if len(names) == 0:
            self.label = None
            return

        texts = [
            TextArea(
                name,
                textprops=dict(
                    color=color,
                    fontproperties=self.viewoptions.font.mpl_props(),
                    path_effects=[withStroke(linewidth=1.5, foreground="black")],
                ),
            )
            for name, color in zip(names, colors)
        ]

        packer = VPacker(pad=0, sep=5, children=texts)

        self.label = AnchoredOffsetbox(
            "upper left", pad=0.5, borderpad=0, frameon=False, child=packer
        )

        self.ax.add_artist(self.label)

    def drawScalebar(self) -> None:
        if self.scalebar is not None:
            self.scalebar.remove()

        self.scalebar = MetricSizeBar(
            self.ax,
            loc="upper right",
            color=self.viewoptions.font.color,
            font_properties=self.viewoptions.font.mpl_props(),
        )
        self.ax.add_artist(self.scalebar)

    def drawData(
        self, data: np.ndarray, extent: Tuple[float, float, float, float]
    ) -> None:
        if self.image is not None:
            self.image.remove()

        self.image = self.ax.imshow(
            data,
            interpolation=self.viewoptions.image.interpolation,
            extent=extent,
            aspect="equal",
            origin="upper",
        )


class OverlayItemRow(QtWidgets.QWidget):
    closeRequested = QtCore.Signal("QWidget*")
    itemChanged = QtCore.Signal()

    def __init__(
        self,
        label: str,
        vmin: Union[str, float],
        vmax: Union[str, float],
        color: QtGui.QColor,
        color_pickable: bool = False,
        parent: "OverlayRows" = None,
    ):
        super().__init__(parent)

        self.label_name = QtWidgets.QLabel(label)
        self.ledit_vmin = QtWidgets.QLineEdit(str(vmin), self)
        self.ledit_vmin.setValidator(PercentOrDecimalValidator(0.0, 1e99))
        self.ledit_vmin.editingFinished.connect(self.itemChanged)
        self.ledit_vmax = QtWidgets.QLineEdit(str(vmax), self)
        self.ledit_vmax.setValidator(PercentOrDecimalValidator(0.0, 1e99))
        self.ledit_vmax.editingFinished.connect(self.itemChanged)

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
        layout.addWidget(QtWidgets.QLabel("min:"), 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.ledit_vmin, 1)
        layout.addWidget(QtWidgets.QLabel("max:"), 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.ledit_vmax, 1)
        layout.addWidget(self.button_hide, 0)
        layout.addWidget(self.button_color, 0)
        layout.addWidget(self.button_close, 0, QtCore.Qt.AlignRight)
        self.setLayout(layout)

    @property
    def hidden(self) -> None:
        return self.action_hide.isChecked()

    def hideChanged(self) -> None:
        if self.hidden:
            self.button_hide.setIcon(QtGui.QIcon.fromTheme("hint"))
        else:
            self.button_hide.setIcon(QtGui.QIcon.fromTheme("visibility"))
        self.itemChanged.emit()

    def getVmin(self, data: np.ndarray) -> float:
        vmin = self.ledit_vmin.text()
        try:
            return float(vmin)
        except ValueError:  # pragma: no cover
            return np.nanpercentile(data, float(vmin.rstrip("%")))

    def getVmax(self, data: np.ndarray) -> float:
        vmax = self.ledit_vmax.text()
        try:
            return float(vmax)
        except ValueError:  # pragma: no cover
            return np.nanpercentile(data, float(vmax.rstrip("%")))

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
        self, label: str, vmin: Union[str, float], vmax: Union[str, float]
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
        row.ledit_vmin.setFocus()
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


class OverlayColocalisationDialog(QtWidgets.QDialog):
    def __init__(self, parent: OverlayTool):
        super().__init__(parent)
        self.widget = parent


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

        path = os.path.join(
            os.path.dirname(self.widget.widget.laser.path),
            self.widget.widget.laser.name + ".png",
        )
        self.lineedit_directory.setText(os.path.dirname(path))
        self.lineedit_filename.setText(os.path.basename(path))
        self.typeChanged(0)

    def isIndividual(self) -> bool:
        return self.check_individual.isChecked() and self.check_individual.isEnabled()

    def updatePreview(self) -> None:
        base, ext = os.path.splitext(self.lineedit_filename.text())
        if self.isIndividual():
            if self.widget.rows.color_model == "rgb":
                base += "_<rgb>"
            elif self.widget.rows.color_model == "cmyk":
                base += "_<cmy>"
            else:
                base += "_<#>"
        self.lineedit_preview.setText(base + ext)

    def filenameChanged(self, filename: str) -> None:
        _, ext = os.path.splitext(filename.lower())
        index = self.options.indexForExt(ext)
        if index == -1:
            return
        self.options.setCurrentIndex(index)
        self.updatePreview()

    def selectDirectory(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(self, "Select Directory", "")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.lineedit_directory.setText)
        dlg.open()
        return dlg

    def getPath(self) -> str:
        return os.path.join(
            self.lineedit_directory.text(), self.lineedit_filename.text()
        )

    def getPathForRow(self, row: int) -> str:
        color_model = self.widget.rows.color_model
        if color_model == "rgb":
            r, g, b, _a = self.widget.rows[row].getColor().getRgb()
            suffix = "r" if r > 0 else "g" if g > 0 else "b"
        elif color_model == "cmyk":
            c, m, y, _k, _a = self.widget.rows[row].getColor().getCmyk()
            suffix = "c" if c > 0 else "m" if m > 0 else "y"
        else:
            suffix = str(row)
        base, ext = os.path.splitext(self.getPath())
        return f"{base}_{suffix}{ext}"

    def generateRowPaths(self) -> Generator[Tuple[str, int], None, None]:
        for i, row in enumerate(self.widget.rows):
            if not row.hidden:
                yield (self.getPathForRow(i), i)

    def export(self, path: str) -> None:
        option = self.options.currentOption()

        if option.ext == ".png":
            self.widget.saveCanvas(path, raw=option.raw())
        else:
            raise io.error.PewException(f"Unable to export file as '{option.ext}'.")

    def exportIndividual(self, path: str, row: int) -> None:
        option = self.options.currentOption()

        if option.ext == ".png":
            data = self.widget.processRow(self.widget.rows[row])
            if self.widget.model_type[self.widget.rows.color_model] == "subtractive":
                data = np.ones_like(data) - data
            if option.raw():
                imsave(path, data)
            else:
                canvas = OverlayCanvas(self.widget.canvas.viewoptions, self)
                canvas.drawData(
                    data, self.widget.widget.laser.config.data_extent(data.shape)
                )
                canvas.view_limits = self.widget.canvas.view_limits

                if canvas.viewoptions.canvas.label:
                    names = [self.widget.rows[row].label_name.text()]
                    canvas.drawLabel(names, ["white"])

                if canvas.viewoptions.canvas.scalebar:
                    canvas.drawScalebar()

                canvas.figure.set_size_inches(
                    self.widget.canvas.figure.get_size_inches()
                )
                canvas.figure.savefig(
                    path, dpi=300, bbox_inches="tight", transparent=True, facecolor=None
                )
                canvas.close()
        else:
            raise io.error.PewException(f"Unable to export file as '{option.ext}'.")

    def accept(self) -> None:
        prompt = OverwriteFilePrompt()
        if self.isIndividual():
            paths = [p for p in self.generateRowPaths() if prompt.promptOverwrite(p[0])]
            if len(paths) == 0:
                return
        else:
            if not prompt.promptOverwrite(self.getPath()):
                return

        try:
            if self.isIndividual():
                for path, row in paths:
                    self.exportIndividual(path, row)
            else:
                self.export(self.getPath())
        except io.error.PewException as e:
            QtWidgets.QMessageBox.critical(self, "Unable to Export!", str(e))
            return

        super().accept()
