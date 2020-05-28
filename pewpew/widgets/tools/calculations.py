import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.lib.calc import normalise, otsu
from pewpew.widgets.ext import ValidColorLineEdit
from pewpew.lib.pratt import Parser, ParserException, Reducer, ReducerException
from pewpew.lib.pratt import BinaryFunction, UnaryFunction, TernaryFunction

from pewpew.widgets.canvases import LaserCanvas
from pewpew.widgets.laser import LaserWidget
from pewpew.widgets.tools import ToolWidget

from typing import List, Union

additional_parser_functions = {
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
additional_reducer_functions = {
    "mean": (np.nanmean, 1),
    "median": (np.nanmedian, 1),
    "normalise": (normalise, 3),
    "otsu": (otsu, 1),
    "percentile": (np.nanpercentile, 2),
    "threshold": (lambda x, a: np.where(x > a, x, np.nan), 2),
}


class NameLineEdit(ValidColorLineEdit):
    def __init__(
        self, text: str, badnames: List[str], parent: QtWidgets.QWidget = None
    ):
        super().__init__(text, parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.badchars = " +-=*/\\^<>!()[]"
        self.badnames = badnames
        self._badnames = ["nan", "if", "then", "else"]
        self._badnames.extend(additional_parser_functions.keys())

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


class FormulaLineEdit(ValidColorLineEdit):
    def __init__(
        self, text: str, variables: List[str], parent: QtWidgets.QWidget = None
    ):
        super().__init__(text, parent)
        self.setClearButtonEnabled(True)
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
            self.expr = self.parser.parse(self.text())
        except ParserException:
            self.expr = ""
        self.revalidate()


class CalculationsTool(ToolWidget):
    def __init__(self, widget: LaserWidget):
        super().__init__(widget)
        self.setWindowTitle("Calculator Tool")

        self.button_apply = QtWidgets.QPushButton("Apply")
        self.button_apply.pressed.connect(self.apply)
        # self.button_apply_all = QtWidgets.QPushButton("Apply To All")
        # self.button_apply_all.pressed.connect(self.applyAll)

        self.canvas = LaserCanvas(self.viewspace.options, parent=self)
        self.output = QtWidgets.QLineEdit("Result")
        self.output.setEnabled(False)
        # self.output.setSizePolicy(
        #     QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum
        # )

        self.lineedit_name = NameLineEdit("", badnames=[])
        self.lineedit_name.revalidate()
        self.lineedit_name.textChanged.connect(self.completeChanged)
        self.lineedit_name.editingFinished.connect(self.refresh)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.activated.connect(self.insertVariable)

        functions = [k + v[1] for k, v in additional_parser_functions.items()]
        tooltips = [v[2] for v in additional_parser_functions.values()]
        self.combo_functions = QtWidgets.QComboBox()
        self.combo_functions.addItem("Functions")
        self.combo_functions.addItems(functions)
        for i in range(0, len(tooltips)):
            self.combo_functions.setItemData(i + 1, tooltips[i], QtCore.Qt.ToolTipRole)
        self.combo_functions.activated.connect(self.insertFunction)

        self.reducer = Reducer({})
        self.result: Union[float, np.ndarray] = None
        self.formula = FormulaLineEdit("", variables=[])
        self.formula.textChanged.connect(self.refresh)
        self.formula.textChanged.connect(self.completeChanged)

        self.reducer.operations.update(additional_reducer_functions)
        self.formula.parser.nulls.update(
            {k: v[0] for k, v in additional_parser_functions.items()}
        )

        layout_combos = QtWidgets.QHBoxLayout()
        layout_combos.addWidget(self.combo_isotope)
        layout_combos.addWidget(self.combo_functions)

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Name:", self.lineedit_name)
        layout_form.addRow("Insert:", layout_combos)
        layout_form.addRow("Formula:", self.formula)
        layout_form.addRow("Result:", self.output)

        self.layout_main.addWidget(self.canvas)
        self.layout_main.addLayout(layout_form)

        self.layout_buttons.addWidget(self.button_apply, 0, QtCore.Qt.AlignRight)

        self.widgetChanged()

    def apply(self) -> None:
        self.widget.laser.add(self.lineedit_name.text(), np.array(self.result))
        self.widget.populateIsotopes()
        self.widgetChanged()

    def insertFunction(self, index: int) -> None:
        if index == 0:
            return
        function = self.combo_functions.currentText()
        function = function[: function.find("(") + 1]
        self.formula.insert(function)
        self.combo_functions.setCurrentIndex(0)
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
        name = self.lineedit_name.text()
        if name == "" or " " in name or name in self.widget.laser.isotopes:
            return False
        return True

    @QtCore.Slot()
    def completeChanged(self) -> None:
        enabled = self.isComplete()
        self.button_apply.setEnabled(enabled)

    def refresh(self) -> None:
        if self.formula.expr == "":
            self.result = None
            self.output.clear()
        else:
            try:
                self.result = self.reducer.reduce(self.formula.expr)
            except (ReducerException, ValueError) as e:
                self.result = None
                self.output.setText(str(e))
                self.formula.setValid(False)
                return

            if np.isscalar(self.result):
                self.output.setText(f"{self.result:.10g}")
            elif isinstance(self.result, np.ndarray):
                self.output.clear()
                extent = self.widget.laser.config.data_extent(self.result.shape)

                self.canvas.drawData(self.result, extent)
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

    def widgetChanged(self) -> None:
        self.label_current.setText(self.widget.laser.name)

        self.combo_isotope.clear()
        self.combo_isotope.addItem("Isotopes")
        self.combo_isotope.addItems(self.widget.laser.isotopes)

        self.lineedit_name.badnames = self.widget.laser.isotopes

        self.reducer.variables = {
            name: self.widget.laser.get(name, calibrate=True, flat=True)
            for name in self.widget.laser.isotopes
        }
        self.formula.parser.variables = self.widget.laser.isotopes
        self.formula.valid = True
        self.formula.setText(self.widget.combo_isotope.currentText())

        self.refresh()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.MouseButtonDblClick and isinstance(
            obj.parent(), LaserWidget
        ):
            self.widget = obj.parent()
            self.endMouseSelect()
        return False
