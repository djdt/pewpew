import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.lib.pratt import Parser, Reducer, Expr, ParserException, BinaryFunction
from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import LaserCanvas
from pewpew.widgets.docks import LaserImageDock
from pewpew.widgets.tools import Tool

from typing import List


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
        self._badnames = ["if", "then", "else", "percentile"]

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
        self.expr: Expr = None

        self.cgood = self.palette().color(QtGui.QPalette.Base)
        self.cbad = QtGui.QColor.fromRgb(255, 172, 172)

    def hasAcceptableInput(self) -> bool:
        return self.expr is not None

    def calculate(self) -> None:
        try:
            self.expr = self.parser.parse(self.text())
        except ParserException:
            self.expr = None
        self.revalidate()


class CalculationsTool(Tool):
    def __init__(
        self,
        dock: LaserImageDock,
        viewoptions: ViewOptions,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.dock = dock
        # Custom viewoptions
        self.viewoptions = ViewOptions()
        self.viewoptions.canvas.colorbar = False
        self.viewoptions.canvas.label = False
        self.viewoptions.canvas.scalebar = False
        self.viewoptions.image.cmap = viewoptions.image.cmap

        self.canvas = LaserCanvas(self.viewoptions)

        self.lineedit_name = NameLineEdit("", badnames=[])
        self.lineedit_name.revalidate()
        self.lineedit_name.textChanged.connect(self.completeChanged)

        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.activated.connect(self.insertVariable)

        self.reducer = Reducer({})
        self.formula = FormulaLineEdit("", variables=[])
        self.formula.textChanged.connect(self.updateCanvas)
        self.formula.textChanged.connect(self.completeChanged)

        self.reducer.operations.update({"percentile": np.percentile})
        self.formula.parser.nulls.update({"percentile": BinaryFunction("percentile")})

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Name:", self.lineedit_name)
        layout_form.addRow("Insert:", self.combo_isotopes)
        layout_form.addRow("Formula:", self.formula)

        self.layout_main.addWidget(self.canvas)
        self.layout_main.addLayout(layout_form)

        self.newDockAdded()

    def apply(self) -> None:
        self.dock.laser.add(
            self.lineedit_name.text(), np.array(self.canvas.image.get_array())
        )
        self.newDockAdded()

    def insertVariable(self, index: int) -> None:
        if index == 0:
            return
        self.formula.setText(
            self.formula.text() + " " + self.combo_isotopes.currentText()
        )
        self.combo_isotopes.setCurrentIndex(0)

    def isComplete(self) -> bool:
        if not self.formula.hasAcceptableInput():
            return False
        name = self.lineedit_name.text()
        if name == "" or " " in name or name in self.dock.laser.isotopes:
            return False
        return True

    def updateCanvas(self) -> None:
        if self.formula.expr is None:
            return
        result = self.reducer.reduce(self.formula.expr)
        if isinstance(result, float):
            return
        # Remove all nan and inf values
        result = np.where(np.isfinite(result), result, 0.0)
        extent = self.dock.laser.config.data_extent(result)
        self.canvas.drawData(result, extent)
        self.canvas.draw()

    def newDockAdded(self) -> None:
        self.combo_isotopes.clear()
        self.combo_isotopes.addItem("Isotopes")
        self.combo_isotopes.addItems(self.dock.laser.isotopes)

        self.lineedit_name.badnames = self.dock.laser.isotopes

        self.reducer.variables = {k: v.data for k, v in self.dock.laser.data.items()}
        self.formula.parser.variables = list(self.dock.laser.isotopes)
        self.formula.valid = True
        self.formula.setText(self.dock.laser.isotopes[0])

        self.updateCanvas()

    @QtCore.Slot("QWidget*")
    def endMouseSelect(self, widget: QtWidgets.QWidget) -> None:
        if self.dock == widget:
            return
        self.dock = widget
        self.newDockAdded()
        super().endMouseSelect(widget)
