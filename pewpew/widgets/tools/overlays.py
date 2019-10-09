import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.image import AxesImage, imsave
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patheffects import withStroke

from pewpew.actions import qAction
from pewpew.validators import PercentOrDecimalValidator

from pewpew.lib.calc import greyscale_to_rgb, normalise
from pewpew.lib.mpltools import MetricSizeBar
from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import BasicCanvas, LaserCanvas
from pewpew.widgets.laser import LaserWidget
from pewpew.widgets.tools import ToolWidget

from typing import List, Tuple, Union


class OverlayTool(ToolWidget):
    model_type = {"any": "additive", "cmyk": "subtractive", "rgb": "additive"}

    def __init__(self, widget: LaserWidget):
        super().__init__(widget)
        self.setWindowTitle("Image Overlay Tool")

        self.button_save = QtWidgets.QPushButton("Save Image")
        self.button_save.pressed.connect(self.openSaveDialog)

        self.canvas = OverlayCanvas(self.viewspace.options)

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

        self.layout_buttons.addWidget(self.button_save, 0, QtCore.Qt.AlignRight)

        self.widgetChanged()

    def isComplete(self) -> bool:
        return self.rows.rowCount() > 0

    @QtCore.Slot()
    def completeChanged(self) -> None:
        enabled = self.isComplete()
        self.button_save.setEnabled(enabled)

    def openSaveDialog(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(
            self,
            "Save File",
            self.widget.laser.path,
            "JPEG images(*.jpg *.jpeg);;PNG images(*.png);;All files(*)",
        )
        dlg.selectNameFilter("PNG images(*.png)")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.fileSelected.connect(self.saveCanvas)
        dlg.open()

    def comboAdd(self, index: int) -> None:
        if index == 0:
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
        datas = [self.processRow(row) for row in self.rows.rows if not row.hidden]
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

        self.canvas.drawData(img, self.widget.laser.config.data_extent(img.shape))

        if self.viewspace.options.canvas.label:
            names = [row.label_name.text() for row in self.rows.rows if not row.hidden]
            colors = [row.getColor().name() for row in self.rows.rows if not row.hidden]
            self.canvas.drawLabel(names, colors)
        elif self.canvas.label is not None:
            self.canvas.label.remove()
            self.canvas.label = None

        if self.viewspace.options.canvas.scalebar:
            self.canvas.drawScalebar()
        elif self.canvas.scalebar is not None:
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

    def widgetChanged(self) -> None:
        self.label_current.setText(self.widget.laser.name)
        self.completeChanged()
        self.refresh()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.MouseButtonDblClick and isinstance(
            obj, LaserCanvas
        ):
            self.widget = obj.parent()
            self.endMouseSelect()
        return False

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        if self.canvas.underMouse():
            action_copy_image = QtWidgets.QAction(
                QtGui.QIcon.fromTheme("insert-image"), "Copy Image", self
            )
            action_copy_image.setStatusTip("Copy image to clipboard.")
            action_copy_image.triggered.connect(self.canvas.copyToClipboard)

            menu = QtWidgets.QMenu(self)
            menu.addAction(action_copy_image)
            menu.popup(event.globalPos())


class OverlayCanvas(BasicCanvas):
    def __init__(self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None):
        super().__init__()
        self.viewoptions = viewoptions

        self.label: AnchoredOffsetbox = None
        self.scalebar: MetricSizeBar = None
        self.image: AxesImage = None

        self.redrawFigure()

    def redrawFigure(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(facecolor="black", autoscale_on=True)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

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
            "visibility", "Hide", "Toggle visibility of this isotope.", self.itemChanged
        )
        self.action_hide.setCheckable(True)

        self.button_hide = QtWidgets.QToolButton(self)
        self.button_hide.setDefaultAction(self.action_hide)

        self.button_color = QtWidgets.QToolButton(self)
        self.button_color.setDefaultAction(self.action_color)
        self.button_color.setEnabled(color_pickable)
        self.setColor(color)

        self.button_close = QtWidgets.QToolButton(self)
        self.button_close.setDefaultAction(self.action_close)

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

    def getVmin(self, data: np.ndarray) -> float:
        vmin = self.ledit_vmin.text()
        try:
            return float(vmin)
        except ValueError:
            return np.nanpercentile(data, float(vmin.rstrip("%")))

    def getVmax(self, data: np.ndarray) -> float:
        vmax = self.ledit_vmax.text()
        try:
            return float(vmax)
        except ValueError:
            return np.nanpercentile(data, float(vmax.rstrip("%")))

    def getColor(self) -> QtGui.QColor:
        return self.button_color.palette().color(QtGui.QPalette.Button)

    def setColor(self, color: QtGui.QColor) -> None:
        palette = self.button_color.palette()
        palette.setColor(QtGui.QPalette.Button, color)
        self.button_color.setPalette(palette)
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
