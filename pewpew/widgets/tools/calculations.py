import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.lib.calc import otsu
from pewpew.lib.pratt import Parser, ParserException, Reducer, ReducerException
from pewpew.lib.pratt import BinaryFunction, UnaryFunction
from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import LaserCanvas
from pewpew.widgets.laser import LaserWidget
from pewpew.widgets.tools import Tool

from typing import List

additional_parser_functions = {
    "mean": UnaryFunction("mean"),
    "median": UnaryFunction("median"),
    "otsu": UnaryFunction("otsu"),
    "percentile": BinaryFunction("percentile"),
    "threshold": BinaryFunction("threshold"),
}
additional_reducer_functions = {
    "mean": (np.nanmean, 1),
    "median": (np.nanmedian, 1),
    "otsu": (otsu, 1),
    "percentile": (np.nanpercentile, 2),
    "threshold": (lambda x, a: np.where(x > a, x, np.nan), 2),
}


class ValidColorLineEdit(QtWidgets.QLineEdit):
    def __init__(self, text: str, parent: QtWidgets.QWidget = None):
        super().__init__(text, parent)
        self.textChanged.connect(self.revalidate)
        self.color_good = self.palette().color(QtGui.QPalette.Base)
        self.color_bad = QtGui.QColor.fromRgb(255, 172, 172)

    def revalidate(self) -> None:
        palette = self.palette()
        if self.hasAcceptableInput():
            color = self.color_good
        else:
            color = self.color_bad
        palette.setColor(QtGui.QPalette.Base, color)
        self.setPalette(palette)


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


class CalculationsTool(Tool):
    def __init__(self, widget: LaserWidget, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Calculator")
        self.widget = widget

        # Custom viewoptions
        self.viewoptions = ViewOptions()
        self.viewoptions.canvas.colorbar = False
        self.viewoptions.canvas.label = False
        self.viewoptions.canvas.scalebar = False
        self.viewoptions.image.cmap = widget.canvas.viewoptions.image.cmap

        self.canvas = LaserCanvas(self.viewoptions)
        self.output = QtWidgets.QLineEdit("Result")
        self.output.setEnabled(False)
        self.output.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding
        )

        self.lineedit_name = NameLineEdit("", badnames=[])
        self.lineedit_name.revalidate()
        self.lineedit_name.textChanged.connect(self.completeChanged)

        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.activated.connect(self.insertVariable)

        self.combo_functions = QtWidgets.QComboBox()
        self.combo_functions.addItem("Functions")
        self.combo_functions.addItems(list(additional_parser_functions.keys()))
        self.combo_functions.activated.connect(self.insertFunction)

        self.reducer = Reducer({})
        self.result = None
        self.formula = FormulaLineEdit("", variables=[])
        self.formula.textChanged.connect(self.updateCanvas)
        self.formula.textChanged.connect(self.completeChanged)

        self.reducer.operations.update(additional_reducer_functions)
        self.formula.parser.nulls.update(additional_parser_functions)

        layout_combos = QtWidgets.QHBoxLayout()
        layout_combos.addWidget(self.combo_isotopes)
        layout_combos.addWidget(self.combo_functions)

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Name:", self.lineedit_name)
        layout_form.addRow("Insert:", layout_combos)
        layout_form.addRow("Formula:", self.formula)
        layout_form.addRow("Result:", self.output)

        self.layout_main.addWidget(self.canvas)
        self.layout_main.addLayout(layout_form)

        self.widgetChanged()

    def apply(self) -> None:
        self.widget.laser.add(self.lineedit_name.text(), np.array(self.result))
        self.widget.populateIsotopes()
        self.widgetChanged()

    def insertFunction(self, index: int) -> None:
        if index == 0:
            return
        text = self.formula.text()
        if text != "" and text[-1] not in " (":
            text += " "
        text += self.combo_functions.currentText() + "("
        self.formula.setText(text)
        self.combo_functions.setCurrentIndex(0)
        self.formula.setFocus()

    def insertVariable(self, index: int) -> None:
        if index == 0:
            return
        text = self.formula.text()
        if text != "" and text[-1] not in " (":
            text += " "
        text += self.combo_isotopes.currentText()
        self.formula.setText(text)
        self.combo_isotopes.setCurrentIndex(0)
        self.formula.setFocus()

    def isComplete(self) -> bool:
        if not self.formula.hasAcceptableInput():
            return False
        name = self.lineedit_name.text()
        if name == "" or " " in name or name in self.widget.laser.isotopes:
            return False
        return True

    def updateCanvas(self) -> None:
        try:
            self.result = self.reducer.reduce(self.formula.expr)
        except ReducerException:
            self.result = None
            self.output.clear()
            return
        if isinstance(self.result, float):
            self.output.setText(f"{self.result:.10g}")
        else:
            self.output.clear()
            extent = self.widget.laser.config.data_extent(self.result)
            self.canvas.drawData(self.result, extent)
            self.canvas.draw()

    def widgetChanged(self) -> None:
        self.combo_isotopes.clear()
        self.combo_isotopes.addItem("Isotopes")
        self.combo_isotopes.addItems(self.widget.laser.isotopes)

        self.lineedit_name.badnames = self.widget.laser.isotopes

        self.reducer.variables = {k: v.data for k, v in self.widget.laser.data.items()}
        self.formula.parser.variables = self.widget.laser.isotopes
        self.formula.valid = True
        self.formula.setText(self.widget.combo_isotopes.currentText())

        self.updateCanvas()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.MouseButtonDblClick and isinstance(
            obj, LaserCanvas
        ):
            self.widget = obj.parent()
            self.widgetChanged()
            self.endMouseSelect()
        return False
