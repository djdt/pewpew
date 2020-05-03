# This tool will replace the calibration tool and the edit / transform options that currently exist.
# Design:
# --------------------|
# |Method  |----------|
# |Options |          |
# | |~~~~~ |          |
# | ~~~~~~ |----------|
# |        |Trans. Ctr|
# |-------------------|

# Methods : Calculator, Blur (Convolve), Deconvolve, Resize / transform

# Options : Calculator - Will require Name, Isotope / Function select, Result, Input
# Blur - Kernel selection, size,
# Deconv: maybe combine with blur?? as conv
# FIltering

import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.lib.calc import normalise, otsu
from pewpew.lib.pratt import Parser, ParserException, Reducer, ReducerException
from pewpew.lib.pratt import BinaryFunction, UnaryFunction, TernaryFunction

from pewpew.widgets.canvases import LaserCanvas
from pewpew.widgets.ext import ValidColorLineEdit, ValidColorTextEdit
from pewpew.widgets.laser import LaserWidget
from pewpew.widgets.tools import ToolWidget

from typing import List, Tuple, Union


class EditTool(ToolWidget):
    METHODS = ["Calculator", "Convole", "Deconvolve", "Filter", "Transform"]

    def __init__(self, widget: LaserWidget):
        super().__init__(widget)

        self.setWindowTitle("Calculator Tool")

        self.button_apply = QtWidgets.QPushButton("Apply")
        self.button_apply.pressed.connect(self.apply)
        # self.button_apply_all = QtWidgets.QPushButton("Apply To All")
        # self.button_apply_all.pressed.connect(self.applyAll)

        self.canvas = LaserCanvas(self.viewspace.options)

        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(EditTool.METHODS)

        self.calculator_method = CalculatorMethod(self)
        self.calculator_method.inputChanged.connect(self.refresh)

        self.transform_method = TransformMethod(self)
        self.transform_method.inputChanged.connect(self.refresh)

        self.method_stack = QtWidgets.QStackedWidget()
        self.method_stack.addWidget(self.calculator_method)
        self.method_stack.addWidget(MethodStackWidget(self))
        self.method_stack.addWidget(MethodStackWidget(self))
        self.method_stack.addWidget(MethodStackWidget(self))
        self.method_stack.addWidget(self.transform_method)
        # Make sure to add the stack widgets in right order!
        self.combo_method.currentIndexChanged.connect(self.setCurrentMethod)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.refresh)

        layout_methods = QtWidgets.QVBoxLayout()
        layout_methods.addWidget(self.method_stack)

        canvas_box = QtWidgets.QGroupBox("Preview")
        layout_canvas_box = QtWidgets.QVBoxLayout()
        layout_canvas_box.addWidget(self.canvas)
        canvas_box.setLayout(layout_canvas_box)

        layout_canvas = QtWidgets.QVBoxLayout()
        layout_canvas.addWidget(canvas_box)
        # layout_canvas.addLayout(transform_bar_layout)
        layout_canvas.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignRight)

        self.layout_top.insertWidget(0, self.combo_method, 0, QtCore.Qt.AlignLeft)

        self.layout_main.setDirection(QtWidgets.QBoxLayout.LeftToRight)
        self.layout_main.addLayout(layout_methods, 0)
        self.layout_main.addLayout(layout_canvas, 1)

        self.widgetChanged()

    def apply(self) -> None:
        # self.widget.laser.add(self.lineedit_name.text(), np.array(self.result))
        # self.widget.populateIsotopes()
        self.widgetChanged()

    def previewData(self, isotope: str = None) -> np.ndarray:
        return self.widget.laser.get(isotope, flat=True, calibrated=False)

    def refresh(self) -> None:
        stack: MethodStackWidget = self.method_stack.currentWidget()
        if not stack.isComplete():  # Not ready for update to preview
            return
        isotope = self.combo_isotope.currentText()
        if stack.full_data:
            data = stack.previewData(self.previewData())
        else:
            data = stack.previewData(self.previewData(isotope))
        if data is None:
            return
        self.canvas.drawData(
            data, self.widget.laser.config.data_extent(data.shape), isotope=isotope,
        )
        if self.canvas.viewoptions.canvas.colorbar:
            self.canvas.drawColorbar(self.widget.laser.calibration[isotope].unit)

        if self.canvas.viewoptions.canvas.label:
            self.canvas.drawLabel(isotope)
        elif self.canvas.label is not None:
            self.canvas.label.remove()
            self.canvas.label = None

        if self.canvas.viewoptions.canvas.scalebar:
            self.canvas.drawScalebar()
        elif self.canvas.scalebar is not None:
            self.canvas.scalebar.remove()
            self.canvas.scalebar = None

    def setCurrentMethod(self, method: int) -> None:
        self.method_stack.setCurrentIndex(method)
        self.combo_isotope.setEnabled(not self.method_stack.currentWidget().full_data)
        self.refresh()

    def widgetChanged(self) -> None:
        self.label_current.setText(self.widget.laser.name)
        # Prevent currentIndexChanged being emmited
        self.combo_isotope.blockSignals(True)
        self.combo_isotope.clear()
        self.combo_isotope.addItems(self.widget.laser.isotopes)
        self.combo_isotope.setCurrentText(self.widget.combo_isotopes.currentText())
        self.combo_isotope.blockSignals(False)

        for i in range(self.method_stack.count()):
            self.method_stack.widget(i).initialise()

        self.refresh()


class MethodStackWidget(QtWidgets.QGroupBox):
    inputChanged = QtCore.Signal()

    def __init__(self, parent: EditTool, full_data: bool = False):
        super().__init__("Options", parent)
        self.edit = parent
        self.full_data = full_data

    def initialise(self) -> None:
        pass

    def isComplete(self) -> bool:
        return True

    def previewData(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CalculatorMethod(MethodStackWidget):
    parser_functions = {
        "mean": (UnaryFunction("mean"), "(<array>)", "Returns the mean of the array."),
        "median": (
            UnaryFunction("median"),
            "(<array>)",
            "Returns the median of the array.",
        ),
        "normalise": (
            TernaryFunction("normalise"),
            "(<array>, <min>, <max>)",
            "Normalise the array from from <min> to <max>.",
        ),
        "otsu": (
            UnaryFunction("otsu"),
            "(<array>)",
            "Returns Otsu's threshold for the array,",
        ),
        "percentile": (
            BinaryFunction("percentile"),
            "(<array>, <percent>)",
            "Returns the <percent> percentile of the array.",
        ),
        "threshold": (
            BinaryFunction("threshold"),
            "(<array>, <value>)",
            "Sets data below <value> to NaN.",
        ),
    }
    reducer_functions = {
        "mean": (np.nanmean, 1),
        "median": (np.nanmedian, 1),
        "normalise": (normalise, 3),
        "otsu": (otsu, 1),
        "percentile": (np.nanpercentile, 2),
        "threshold": (lambda x, a: np.where(x > a, x, np.nan), 2),
    }

    def __init__(self, parent: EditTool):
        super().__init__(parent, full_data=True)
        self.output = QtWidgets.QLineEdit("Result")
        self.output.setEnabled(False)

        self.lineedit_name = CalculatorName(
            "", badnames=[], badparser=list(CalculatorMethod.parser_functions.keys()),
        )
        self.lineedit_name.revalidate()
        self.lineedit_name.textChanged.connect(self.inputChanged)
        # self.lineedit_name.editingFinished.connect(self.refresh)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.activated.connect(self.insertVariable)

        functions = [k + v[1] for k, v in CalculatorMethod.parser_functions.items()]
        tooltips = [v[2] for v in CalculatorMethod.parser_functions.values()]
        self.combo_function = QtWidgets.QComboBox()
        self.combo_function.addItem("Functions")
        self.combo_function.addItems(functions)
        for i in range(0, len(tooltips)):
            self.combo_function.setItemData(i + 1, tooltips[i], QtCore.Qt.ToolTipRole)
        self.combo_function.activated.connect(self.insertFunction)

        self.reducer = Reducer({})
        self.result: Union[float, np.ndarray] = None
        self.formula = CalculatorFormula("", variables=[])
        self.formula.textChanged.connect(self.inputChanged)

        self.reducer.operations.update(CalculatorMethod.reducer_functions)
        self.formula.parser.nulls.update(
            {k: v[0] for k, v in CalculatorMethod.parser_functions.items()}
        )

        layout_combos = QtWidgets.QHBoxLayout()
        layout_combos.addWidget(self.combo_isotope)
        layout_combos.addWidget(self.combo_function)

        # layout_form = QtWidgets.QFormLayout()
        layout_grid = QtWidgets.QGridLayout()
        layout_grid.addWidget(QtWidgets.QLabel("Name:"), 0, 0)
        layout_grid.addWidget(self.lineedit_name, 0, 1)
        layout_grid.addWidget(QtWidgets.QLabel("Insert:"), 1, 0)
        layout_grid.addLayout(layout_combos, 1, 1)
        layout_grid.addWidget(QtWidgets.QLabel("Formula:"), 2, 0)
        layout_grid.addWidget(self.formula, 2, 1, 1, 1)
        layout_grid.addWidget(QtWidgets.QLabel("Result:"), 3, 0)
        layout_grid.addWidget(self.output, 3, 1)

        # layout_grid.setRowStretch(2, 3)
        # layout_form.addRow("Name:", self.lineedit_name, 1)
        # layout_form.addRow("Insert:", layout_combos, 1)
        # layout_form.addRow("Formula:", self.formula, 3)
        # layout_form.addRow("Result:", self.output, 1)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_combos)
        layout_main.addLayout(layout_grid)
        layout_main.addStretch(1)
        self.setLayout(layout_main)

    def initialise(self) -> None:
        isotopes = self.edit.widget.laser.isotopes
        self.combo_isotope.clear()
        self.combo_isotope.addItem("Isotopes")
        self.combo_isotope.addItems(isotopes)

        self.lineedit_name.badnames = isotopes

        # self.reducer.variables = {
        #     name: self.widget.laser.get(name, calibrate=True, flat=True)
        #     for name in self.widget.laser.isotopes
        # }
        self.formula.parser.variables = isotopes
        self.formula.valid = True
        self.formula.setText(self.edit.combo_isotope.currentText())

    def insertFunction(self, index: int) -> None:
        if index == 0:
            return
        function = self.combo_function.currentText()
        function = function[: function.find("(") + 1]
        self.formula.insert(function)
        self.combo_function.setCurrentIndex(0)
        self.formula.setFocus()

    def insertVariable(self, index: int) -> None:
        if index == 0:
            return
        self.formula.insert(self.combo_isotope.currentText())
        self.combo_isotope.setCurrentIndex(0)
        self.formula.setFocus()

    def isComplete(self) -> bool:
        if not self.formula.hasAcceptableInput():
            return False
        if np.isscalar(self.result):
            return False
        if self.lineedit_name.hasAcceptableInput():
            return False
        return True

    def previewData(self, data: np.ndarray) -> np.ndarray:
        self.reducer.variables = {name: data[name] for name in data.dtype.names}
        try:
            data = self.reducer.reduce(self.formula.expr)
            if np.isscalar(data):
                self.output.setText(f"{data:.10g}")
                return None
            elif isinstance(data, np.ndarray):
                return data
        except (ReducerException, ValueError) as e:
            self.output.setText(str(e))
            return None


class CalculatorName(ValidColorLineEdit):
    def __init__(
        self,
        text: str,
        badnames: List[str],
        badparser: List[str],
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(text, parent)

        self.badchars = " +-=*/\\^<>!()[]"
        self.badnames = badnames
        self._badnames = ["nan", "if", "then", "else"]
        self._badnames.extend(badparser)

    def hasAcceptableInput(self) -> bool:
        if self.text() == "":
            return False
        if any(c in self.text() for c in self.badchars):
            return False
        if self.text() in self._badnames:
            return False
        if self.text() in self.badnames:
            return False
        return True


class CalculatorFormula(ValidColorTextEdit):
    def __init__(
        self, text: str, variables: List[str], parent: QtWidgets.QWidget = None
    ):
        super().__init__(text, parent)
        # self.setClearButtonEnabled(True)
        self.textChanged.disconnect(self.revalidate)
        self.textChanged.connect(self.calculate)
        self.parser = Parser(variables)
        self.expr = ""

        self.cgood = self.palette().color(QtGui.QPalette.Base)
        self.cbad = QtGui.QColor.fromRgb(255, 172, 172)

    def hasAcceptableInput(self) -> bool:
        return self.expr != ""

    def calculate(self) -> None:
        try:
            self.expr = self.parser.parse(self.toPlainText())
        except ParserException:
            self.expr = ""
        self.revalidate()


class TransformMethod(MethodStackWidget):
    def __init__(self, parent: EditTool):
        super().__init__(parent)

        self.lineedit_scale = QtWidgets.QLineEdit("1")
        self.lineedit_scale.setValidator(QtGui.QIntValidator(1, 10))
        self.lineedit_scale.textChanged.connect(self.inputChanged)

        self.lineedit_trim_x1 = QtWidgets.QLineEdit("0")
        self.lineedit_trim_x1.setValidator(QtGui.QIntValidator(0, 9999))
        self.lineedit_trim_x1.textChanged.connect(self.inputChanged)
        self.lineedit_trim_x2 = QtWidgets.QLineEdit("0")
        self.lineedit_trim_x2.setValidator(QtGui.QIntValidator(0, 9999))
        self.lineedit_trim_x2.textChanged.connect(self.inputChanged)

        self.lineedit_trim_y1 = QtWidgets.QLineEdit("0")
        self.lineedit_trim_y1.setValidator(QtGui.QIntValidator(0, 9999))
        self.lineedit_trim_y1.textChanged.connect(self.inputChanged)
        self.lineedit_trim_y2 = QtWidgets.QLineEdit("0")
        self.lineedit_trim_y2.setValidator(QtGui.QIntValidator(0, 9999))
        self.lineedit_trim_y2.textChanged.connect(self.inputChanged)

        layout_trim_x = QtWidgets.QHBoxLayout()
        layout_trim_x.addWidget(self.lineedit_trim_x1)
        layout_trim_x.addWidget(QtWidgets.QLabel("x"))
        layout_trim_x.addWidget(self.lineedit_trim_x2)
        layout_trim_y = QtWidgets.QHBoxLayout()
        layout_trim_y.addWidget(self.lineedit_trim_y1)
        layout_trim_y.addWidget(QtWidgets.QLabel("x"))
        layout_trim_y.addWidget(self.lineedit_trim_y2)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Scale", self.lineedit_scale)
        layout.addRow("Trim X", layout_trim_x)
        layout.addRow("Trim Y", layout_trim_y)

        self.setLayout(layout)

    @property
    def scale(self) -> int:
        return int(self.lineedit_scale.text())

    @property
    def trim(self) -> Tuple[int, int, int, int]:
        return (
            int(self.lineedit_trim_x1.text()),
            int(self.lineedit_trim_x2.text()),
            int(self.lineedit_trim_y1.text()),
            int(self.lineedit_trim_y2.text()),
        )

    def initialise(self) -> None:
        self.lineedit_scale.setText("1")
        self.lineedit_trim_x1.setText("0")
        self.lineedit_trim_x2.setText("0")
        self.lineedit_trim_y1.setText("0")
        self.lineedit_trim_y2.setText("0")

    def isComplete(self) -> bool:
        trim = self.trim
        if not self.lineedit_scale.hasAcceptableInput():
            return False
        if not all(
            le.hasAcceptableInput()
            for le in [
                self.lineedit_trim_x1,
                self.lineedit_trim_x2,
                self.lineedit_trim_y1,
                self.lineedit_trim_y2,
            ]
        ):
            return False
        if trim[2] + trim[3] > self.edit.widget.laser.shape[0] * self.scale:
            return False
        if trim[0] + trim[1] > self.edit.widget.laser.shape[1] * self.scale:
            return False
        return True

    def previewData(self, data: np.ndarray) -> None:
        data = np.repeat(np.repeat(data, self.scale, axis=0), self.scale, axis=1)
        trim = self.trim
        return data[
            trim[2] : data.shape[0] - trim[3], trim[0] : data.shape[1] - trim[1],
        ]
