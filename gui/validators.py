from PyQt5 import QtGui, QtWidgets

from typing import Tuple


class PercentValidator(QtGui.QValidator):
    def __init__(
        self,
        percent_min: int = 0,
        percent_max: int = 100,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.percent_min = percent_min
        self.percent_max = percent_max

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

        if i < self.percent_min:
            return (QtGui.QValidator.Intermediate, input, pos)
        elif i > self.percent_max:
            return (QtGui.QValidator.Invalid, input, pos)

        return (QtGui.QValidator.Acceptable, input, pos)


class PercentOrIntValidator(PercentValidator):
    def __init__(
        self,
        int_min: int = None,
        int_max: int = None,
        percent_min: int = 0,
        percent_max: int = 100,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(percent_min, percent_max, parent)
        self.int_min = int_min
        self.int_max = int_max

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        if len(input) == 0:
            return (QtGui.QValidator.Intermediate, input, pos)

        # Treat as percent
        if "%" in input:
            return super().validate(input, pos)

        # Treat as int
        try:
            i = int(input)
        except ValueError:
            return (QtGui.QValidator.Invalid, input, pos)

        if self.int_min is not None and i < self.int_min:
            return (QtGui.QValidator.Intermediate, input, pos)
        elif self.int_max is not None and i > self.int_max:
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
