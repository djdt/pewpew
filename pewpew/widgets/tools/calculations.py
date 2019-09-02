from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.lib.calc import FormulaException, FormulaParser

class FormulaLineEdit(QtWidgets.QLineEdit):
    def __init__(self, text: str, variables: dict, parent: QtWidgets.QWidget = None):
        super().__init__(text, parent)

        self.parser = FormulaParser(variables)
        self.textEdited.connect(self.updateParser)

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

    def updateParser(self) -> None:
        self.valid = self.parser.validateString(self.text())

    def hasAcceptableInput(self) -> bool:
        return self.valid
