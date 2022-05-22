import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib.process import filters

from pewpew.actions import qAction, qToolButton
from pewpew.graphics import colortable
from pewpew.graphics.imageitems import LaserImageItem, ScaledImageItem

from pewpew.widgets.ext import ValidColorLineEdit
from pewpew.widgets.tools import ToolWidget

from pewpew.validators import ConditionalLimitValidator

from typing import Callable, List, Optional, Tuple

from pewpew.widgets.views import TabView


# Filters
def rolling_mean(x: np.ndarray, size: int, threshold: float) -> np.ndarray:
    """For mapping size to (size, size)."""
    size = int(size)
    return filters.rolling_mean(x, (size, size), threshold)


def rolling_median(x: np.ndarray, size: int, threshold: float) -> np.ndarray:
    """For mapping size to (size, size)."""
    size = int(size)
    return filters.rolling_median(x, (size, size), threshold)


# def simple_highpass(x: np.ndarray, limit: float, replace: float) -> np.ndarray:
#     return np.where(x < limit, replace, x)


# def simple_lowpass(x: np.ndarray, limit: float, replace: float) -> np.ndarray:
#     return np.where(x > limit, replace, x)


class FilteringTool(ToolWidget):
    """View and calculate mean and meidan filtered images."""

    methods: dict = {
        "Rolling Mean": {
            "filter": rolling_mean,
            "params": [
                ("size", 5, (2.5, 99), lambda x: (x + 1) % 2 == 0),
                ("σ", 3.0, (0.0, np.inf), None),
            ],
            "desc": ["Window size for local mean.", "Filter if > σ stddevs from mean."],
        },
        "Rolling Median": {
            "filter": rolling_median,
            "params": [
                ("size", 5, (2.5, 99), lambda x: (x + 1) % 2 == 0),
                ("k", 3.0, (0.0, np.inf), None),
            ],
            "desc": [
                "Window size for local median.",
                "Filter if median absolute deviation > k stddevs",
            ],
        },
        # "Simple High-pass": {
        #     "filter": simple_highpass,
        #     "params": [
        #         ("min", 1e3, (-np.inf, np.inf), None),
        #         ("replace", 0.0, (-np.inf, np.inf), None),
        #     ],
        #     "desc": ["Filter if below this value.", "Value to replace with."],
        # },
        # "Simple Low-pass": {
        #     "filter": simple_lowpass,
        #     "params": [
        #         ("max", 1e3, (-np.inf, np.inf), None),
        #         ("replace", 0.0, (-np.inf, np.inf), None),
        #     ],
        #     "desc": ["Filter if above this value.", "Value to replace with."],
        # },
    }

    def __init__(self, item: LaserImageItem, view: Optional[TabView] = None):
        super().__init__(item, graphics_label="Preview", view=view)

        self.image: Optional[ScaledImageItem] = None

        self.action_toggle_filter = qAction(
            "visibility",
            "Filter Visible",
            "Toggle visibility of the filtering.",
            self.toggleFilter,
        )
        self.action_toggle_filter.setCheckable(True)
        self.button_hide_filter = qToolButton(action=self.action_toggle_filter)
        self.button_hide_filter.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.activated.connect(self.completeChanged)
        self.combo_element.activated.connect(self.refresh)

        self.box_graphics.layout().addWidget(
            self.combo_element, 0, QtCore.Qt.AlignRight
        )

        self.combo_filter = QtWidgets.QComboBox()
        self.combo_filter.addItems(FilteringTool.methods.keys())
        self.combo_filter.setCurrentText("Rolling Median")
        self.combo_filter.activated.connect(self.filterChanged)
        self.combo_filter.activated.connect(self.completeChanged)
        self.combo_filter.activated.connect(self.refresh)

        nparams = np.amax([len(f["params"]) for f in FilteringTool.methods.values()])
        self.label_fparams = [QtWidgets.QLabel() for _ in range(nparams)]
        self.lineedit_fparams = [ValidColorLineEdit() for _ in range(nparams)]
        for le in self.lineedit_fparams:
            le.textEdited.connect(self.completeChanged)
            le.editingFinished.connect(self.refresh)
            le.setValidator(ConditionalLimitValidator(0.0, 0.0, 4, condition=None))

        layout_controls = QtWidgets.QFormLayout()
        layout_controls.addWidget(self.combo_filter)
        for i in range(len(self.label_fparams)):
            layout_controls.addRow(self.label_fparams[i], self.lineedit_fparams[i])
        layout_controls.addWidget(self.button_hide_filter)

        self.box_controls.setLayout(layout_controls)

        self.initialise()

    def apply(self) -> None:
        name = self.combo_element.currentText()
        if self.button_hide_filter.isChecked():
            filter_ = FilteringTool.methods[self.combo_filter.currentText()]["filter"]
            self.item.laser.data[name] = filter_(
                self.item.laser.data[name], *self.fparams
            )
        else:
            self.item.laser.data[name] = self.filtered_data

        self.item.redraw()
        self.initialise()

    @property
    def fparams(self) -> List[float]:
        return [float(le.text()) for le in self.lineedit_fparams if le.isEnabled()]

    def filterChanged(self) -> None:
        filter_ = FilteringTool.methods[self.combo_filter.currentText()]
        # Clear all the current params
        for le in self.label_fparams:
            le.setVisible(False)
        for le in self.lineedit_fparams:
            le.setVisible(False)

        params: List[Tuple[str, float, Tuple, Callable[[float], bool]]] = filter_[
            "params"
        ]

        for i, (symbol, default, range, condition) in enumerate(params):
            self.label_fparams[i].setText(f"{symbol}:")
            self.label_fparams[i].setVisible(True)
            self.lineedit_fparams[i].validator().setRange(range[0], range[1], 4)
            self.lineedit_fparams[i].validator().setCondition(condition)
            self.lineedit_fparams[i].setVisible(True)
            self.lineedit_fparams[i].setToolTip(filter_["desc"][i])
            # keep input that's still valid
            if not self.lineedit_fparams[i].hasAcceptableInput():
                self.lineedit_fparams[i].setText(str(default))
                self.lineedit_fparams[i].revalidate()

    def initialise(self) -> None:
        elements = self.item.laser.elements
        self.combo_element.clear()
        self.combo_element.addItems(elements)

        self.filterChanged()
        self.refresh()

    def isComplete(self) -> bool:
        if not all(
            le.hasAcceptableInput() for le in self.lineedit_fparams if le.isEnabled()
        ):
            return False
        return True

    def refresh(self) -> None:
        if not self.isComplete():  # Not ready for update to preview
            return

        element = self.combo_element.currentText()

        data = self.item.laser.get(element, flat=True, calibrated=False)
        if not self.button_hide_filter.isChecked():
            filter_ = FilteringTool.methods[self.combo_filter.currentText()]["filter"]
            data = filter_(data, *self.fparams)
            self.filtered_data = data
        else:
            self.filtered_data = None

        x0, x1, y0, y1 = self.item.laser.config.data_extent(data.shape)
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

        vmin, vmax = self.item.options.get_color_range_as_float("<calc>", data)
        data = np.clip(data, vmin, vmax)
        if vmin != vmax:
            data = (data - vmin) / (vmax - vmin)

        table = colortable.get_table(self.item.options.colortable)

        if self.image is not None:
            self.graphics.scene().removeItem(self.image)

        self.image = ScaledImageItem.fromArray(data, rect, table)
        self.graphics.scene().addItem(self.image)

        self.graphics.colorbar.updateTable(
            table, vmin, vmax, self.item.laser.calibration[element].unit
        )
        self.graphics.invalidateScene()

    def toggleFilter(self, hide: bool) -> None:
        if hide:
            self.button_hide_filter.setIcon(QtGui.QIcon.fromTheme("hint"))
        else:
            self.button_hide_filter.setIcon(QtGui.QIcon.fromTheme("visibility"))

        self.refresh()
