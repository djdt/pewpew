import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.image import AxesImage

from pew.laser import Laser

from pewpew.actions import qAction
from pewpew.validators import PercentOrDecimalValidator

from pewpew.lib.calc import greyscale_to_rgb, normalise
from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import BasicCanvas
from pewpew.widgets.tools import Tool

from typing import List, Tuple, Union


class OverlayCanvas(BasicCanvas):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__()
        self.ax = self.figure.subplots()
        self.ax.set_facecolor("black")
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.autoscale(False)

        self.image: AxesImage = None

    def drawData(
        self,
        data: np.ndarray,
        extent: Tuple[float, float, float, float],
        facecolor: str = "black",
    ) -> None:
        self.ax.clear()
        self.ax.patch.set_facecolor(facecolor)
        self.image = self.ax.imshow(
            data, interpolation="none", extent=extent, aspect="equal", origin="upper"
        )
        self.draw_idle()


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
        if self.rowCount() > self.max_rows:
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


class OverlayTool(Tool):
    model_type = {"any": "additive", "cmyk": "subtractive", "rgb": "additive"}

    def __init__(
        self, laser: Laser, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Image Overlay Tool")
        self.button_box.button(QtWidgets.QDialogButtonBox.Apply).setVisible(False)

        self.laser = laser
        self.viewoptions = viewoptions

        self.canvas = OverlayCanvas()

        self.check_normalise = QtWidgets.QCheckBox("Renormalise")
        self.check_normalise.setEnabled(False)
        self.check_normalise.clicked.connect(self.updateCanvas)

        self.radio_rgb = QtWidgets.QRadioButton("rgb")
        self.radio_rgb.setChecked(True)
        self.radio_rgb.toggled.connect(self.updateColorModel)
        self.radio_cmyk = QtWidgets.QRadioButton("cmyk")
        self.radio_cmyk.toggled.connect(self.updateColorModel)
        self.radio_custom = QtWidgets.QRadioButton("any")
        self.radio_custom.toggled.connect(self.updateColorModel)

        self.combo_add = QtWidgets.QComboBox()
        self.combo_add.addItem("Add Isotope")
        self.combo_add.addItems(laser.isotopes)
        self.combo_add.activated[int].connect(self.comboAdd)

        self.rows = OverlayRows(self)
        self.rows.rowsChanged.connect(self.rowsChanged)
        self.rows.itemChanged.connect(self.updateCanvas)

        layout_top = QtWidgets.QHBoxLayout()
        layout_top.addWidget(self.combo_add, 1, QtCore.Qt.AlignLeft)
        layout_top.addWidget(self.radio_rgb, 0, QtCore.Qt.AlignRight)
        layout_top.addWidget(self.radio_cmyk, 0, QtCore.Qt.AlignRight)
        layout_top.addWidget(self.radio_custom, 0, QtCore.Qt.AlignRight)

        self.layout_main.addWidget(self.canvas, 1)
        self.layout_main.addLayout(layout_top)
        self.layout_main.addWidget(self.rows, 0)
        self.layout_main.addWidget(self.check_normalise, 0)
        # Draw blank
        self.updateCanvas()

    def comboAdd(self, index: int) -> None:
        if index == 0:
            return
        text = self.combo_add.itemText(index)
        self.addRow(text)
        self.combo_add.setCurrentIndex(0)

    def addRow(self, label: str) -> None:
        vmin, vmax = self.viewoptions.colors.get_range(label)
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
        self.updateCanvas()

    def processRow(self, row: OverlayItemRow) -> np.ndarray:
        img = self.laser.get(row.label_name.text(), flat=True)
        vmin, vmax = row.getVmin(img), row.getVmax(img)

        r, g, b, _a = row.getColor().getRgbF()
        if self.model_type[self.rows.color_model] == "subtractive":
            r, g, b = 1.0 - r, 1.0 - g, 1.0 - b

        return greyscale_to_rgb(
            (np.clip(img, vmin, vmax) - vmin) / (vmax - vmin), (r, g, b)
        )

    def updateCanvas(self) -> None:
        datas = [self.processRow(row) for row in self.rows.rows if not row.hidden]
        if len(datas) == 0:
            img = np.zeros((*self.laser.shape[:2], 3), float)
        else:
            img = np.sum(datas, axis=0)

        if self.model_type[self.rows.color_model] == "subtractive":
            img = np.ones_like(img) - img

        if self.check_normalise.isChecked() and self.check_normalise.isEnabled():
            img = normalise(img, 0.0, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)

        self.canvas.drawData(img, self.laser.config.data_extent(img.shape))
