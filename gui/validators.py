from PyQt5 import QtGui, QtWidgets

from typing import Tuple


class PercentValidator(QtGui.QValidator):
    def __init__(
        self, min_value: int = 0, max_value: int = 100, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        if len(input.rstrip("%")) == 0:
            return (QtGui.QValidator.Intermediate, input, pos)

        if not input.endswith("%"):
            input += "%"
        elif input.count("%") > 1:
            return (QtGui.QValidator.Invalid, input, pos)

        try:
            i = int(input.rstrip("%"))
        except ValueError:
            return (QtGui.QValidator.Invalid, input, pos)

        if i < self.min_value:
            return (QtGui.QValidator.Intermediate, input, pos)
        elif i > self.max_value:
            return (QtGui.QValidator.Invalid, input, pos)

        return (QtGui.QValidator.Acceptable, input, pos)


class DoublePrecisionDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, decimals: int, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.decimals = decimals

    # def createEditor(self, parent, option, index):
    #     line_edit = QtWidgets.QLineEdit(parent)
    #     line_edit.setValidator(QtGui.QDoubleValidator(0, 1e99, self.decimals))
    #     return line_edit

    def displayText(self, value: str, locale: str) -> str:
        try:
            num = float(value)
            return f"{num:.{self.decimals}f}"
        except (TypeError, ValueError):
            return str(super().displayText(value, locale))
