import numpy as np

from PySide2 import QtCore, QtWidgets

from pewlib.process import convolve, filters

from pewpew.actions import qAction, qToolButton
from pewpew.lib import kmeans
from pewpew.lib.pratt import Parser, ParserException, Reducer, ReducerException
from pewpew.lib.pratt import BinaryFunction, UnaryFunction, TernaryFunction

from pewpew.widgets.canvases import BasicCanvas, LaserImageCanvas
from pewpew.widgets.ext import ValidColorLineEdit, ValidColorTextEdit
from pewpew.widgets.laser import LaserWidget
from pewpew.widgets.tools import ToolWidget

from pewpew.validators import LimitValidator, OddIntValidator

from typing import List, Tuple


def simple_lowpass(x: np.ndarray, limit: float, replace: float) -> np.ndarray:
    return np.where(x > limit, replace, x)


class FilteringTool(ToolWidget):
    methods: dict = {
        "Simple-LowPass": {
            "filter": simple_lowpass,
            "params": [("max", 0.5, (0.0, 1.0), None)],
            "desc": ["Filter if low pass changes value d amount."],
        },
        "Mean": {
            "filter": filters.rolling_mean,
            "params": [("size", 5, (3, 99), 2), ("σ", 3.0, (0.0, np.inf), None)],
            "desc": ["Filter if > σ stddevs from mean."],
        },
        "Median": {
            "filter": filters.rolling_median,
            "params": [("size", 5, (3, 99), 2), ("M", 3.0, (0.0, np.inf), None)],
            "desc": ["Filter if > M medians from median."],
        },
    }

    def __init__(self, widget: LaserWidget):
        super().__init__(widget, canvas_label="Preview")

        self.combo_filter = QtWidgets.QComboBox()
        self.combo_filter.addItems(FilteringTool.methods.keys())
        self.combo_filter.setCurrentText("Median")
        self.combo_filter.activated.connect(self.filterChanged)
        self.combo_filter.activated.connect(self.inputChanged)

#         self.lineedit_fsize = ValidColorLineEdit()
#         self.lineedit_fsize.setValidator(OddIntValidator(3, 21))
#         self.lineedit_fsize.editingFinished.connect(self.inputChanged)

        nparams = np.amax([len(f["params"]) for f in FilteringTool.methods.values()])
        self.label_fparams = [QtWidgets.QLabel() for i in range(nparams)]
        self.lineedit_fparams = [ValidColorLineEdit() for i in range(nparams)]
        for le in self.lineedit_fparams:
            le.editingFinished.connect(self.inputChanged)
            le.setValidator(LimitValidator(0.0, 0.0, 4, None))

        layout_filter = QtWidgets.QFormLayout()
        layout_filter.addWidget(self.combo_filter)
        layout_filter.addRow("Size:", self.lineedit_fsize)
        for i in range(len(self.label_fparams)):
            layout_filter.addRow(self.label_fparams[i], self.lineedit_fparams[i])

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_filter)

        self.setLayout(layout_main)
        self.filterChanged()

    @property
    def fsize(self) -> int:
        return int(self.lineedit_fsize.text())

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

        params: List[Tuple[str, float, Tuple]] = filter_["params"]

        for i, (symbol, default, range, modulus) in enumerate(params):
            self.label_fparams[i].setText(f"{symbol}:")
            self.label_fparams[i].setVisible(True)
            self.lineedit_fparams[i].validator().setRange(range[0], range[1], 4)
            self.lineedit_fparams[i].validator().setModulus(modulus)
            self.lineedit_fparams[i].setVisible(True)
            self.lineedit_fparams[i].setToolTip(filter_["desc"][i])
            # keep input that's still valid
            if not self.lineedit_fparams[i].hasAcceptableInput():
                self.lineedit_fparams[i].setText(str(default))
                self.lineedit_fparams[i].revalidate()

    def isComplete(self) -> bool:
        if not self.lineedit_fsize.hasAcceptableInput():
            return False
        if not all(
            le.hasAcceptableInput() for le in self.lineedit_fparams if le.isEnabled()
        ):
            return False
        return True

    def previewData(self, data: np.ndarray) -> np.ndarray:
        filter_ = FilteringTool.methods[self.combo_filter.currentText()]["filter"]
        return filter_(data, (self.fsize, self.fsize), *self.fparams)

    def refresh(self) -> None:
        if not self.isComplete():  # Not ready for update to preview
            return

        isotope = self.combo_isotope.currentText()

        data = self.previewData(self.widget.laser.get(flat=True, calibrated=False))
        if data is None:
            return
        extent = self.widget.laser.config.data_extent(data.shape)

        # Only change the view if new or the laser extent has changed (i.e. conf edit)
        if self.canvas.extent != extent:
            self.canvas.view_limits = self.canvas.extentForAspect(extent)

        self.canvas.drawData(
            data,
            extent,
            isotope=isotope,
        )

        if self.canvas.viewoptions.canvas.colorbar:
            self.canvas.drawColorbar("")

        if self.canvas.viewoptions.canvas.label:
            self.canvas.drawLabel(self.lineedit_name.text())
        elif self.canvas.label is not None:
            self.canvas.label.remove()
            self.canvas.label = None

        if self.canvas.viewoptions.canvas.scalebar:
            self.canvas.drawScalebar()
        elif self.canvas.scalebar is not None:
            self.canvas.scalebar.remove()
            self.canvas.scalebar = None

        self.canvas.draw_idle()
