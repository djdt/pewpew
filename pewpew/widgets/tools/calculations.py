import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.lib.pratt import Parser, ParserException
from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import LaserCanvas
from pewpew.widgets.docks import LaserImageDock
from pewpew.widgets.tools import Tool

from typing import Union


class FormulaLineEdit(QtWidgets.QLineEdit):
    def __init__(self, text: str, variables: dict, parent: QtWidgets.QWidget = None):
        super().__init__(text, parent)
        self.setClearButtonEnabled(True)
        self.parser = Parser(variables)

        self._valid = True
        self.cgood = self.palette().color(QtGui.QPalette.Base)
        self.cbad = QtGui.QColor.fromRgb(255, 172, 172)

    @property
    def valid(self) -> bool:
        return self._valid

    @valid.setter
    def valid(self, valid: bool) -> None:
        self._valid = valid
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Base, self.cgood if valid else self.cbad)
        self.setPalette(palette)

    def hasAcceptableInput(self) -> bool:
        return self.valid

    def value(self) -> Union[float, np.ndarray]:
        try:
            result = self.parser.reduceString(self.text())
            self.valid = True
            return result
        except ParserException:
            self.valid = False
            return None


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

        self.formula = FormulaLineEdit("", variables={})
        self.formula.textChanged.connect(self.updateCanvas)
        self.canvas = LaserCanvas(self.viewoptions)

        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.activated.connect(self.insertVariable)

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Insert:", self.combo_isotopes)
        layout_form.addRow("Formula:", self.formula)

        self.layout_center.addWidget(self.canvas)
        self.layout_center.addLayout(layout_form)

        self.newDockAdded()

    def insertVariable(self, index: int) -> None:
        if index == 0:
            return
        self.formula.setText(
            self.formula.text() + " " + self.combo_isotopes.currentText()
        )
        self.combo_isotopes.setCurrentIndex(0)

    def updateCanvas(self) -> None:
        result = self.formula.value()

        if result is None or isinstance(result, float):
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

        self.formula.parser.variables = {
            k: v.data for k, v in self.dock.laser.data.items()
        }
        self.formula.valid = True
        self.formula.setText(self.dock.laser.isotopes[0])

    @QtCore.Slot("QWidget*")
    def mouseSelectFinished(self, widget: QtWidgets.QWidget) -> None:
        if widget is not None and hasattr(widget, "laser"):
            self.dock = widget

        self.dockarea.mouseSelectFinished.disconnect(self.mouseSelectFinished)
        self.activateWindow()
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self.show()
        self.draw()

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() in [
            QtCore.Qt.Key_Escape,
            QtCore.Qt.Key_Enter,
            QtCore.Qt.Key_Return,
        ]:
            return
        if event.key() == QtCore.Qt.Key_F5:
            self.draw()
        super().keyPressEvent(event)
