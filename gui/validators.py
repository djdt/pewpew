from PyQt5 import QtGui, QtWidgets

from typing import Tuple


class DecimalValidator(QtGui.QDoubleValidator):
    def __init__(
        self, bottom: float, top: float, decimals: int, parent: QtWidgets.QWidget = None
    ):
        super().__init__(bottom, top, decimals, parent)
        self.setNotation(QtGui.QDoubleValidator.StandardNotation)


class PercentOrDecimalValidator(DecimalValidator):
    def __init__(
        self,
        bottom: float = -1e10,
        top: float = 1e10,
        decimals: int = 4,
        percent_bottom: float = 0.0,
        percent_top: float = 100.0,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(percent_bottom, percent_top, decimals, parent)
        self.bottom = bottom
        self.top = top
        self.decimals = decimals
        self.percent_bottom = percent_bottom
        self.percent_top = percent_top

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        # Treat as percent
        if "%" in input:
            if not input.endswith("%") or input.count("%") > 1:
                return (QtGui.QValidator.Invalid, input, pos)
            self.setRange(self.percent_bottom, self.percent_top, self.decimals)
            return (super().validate(input.rstrip("%"), pos)[0], input, pos)
        # Treat as double
        self.setRange(self.bottom, self.top, self.decimals)
        return super().validate(input, pos)


# class PercentOrIntValidator(PercentValidator):
#     def __init__(
#         self,
#         int_min: int = None,
#         int_max: int = None,
#         percent_min: int = 0,
#         percent_max: int = 100,
#         parent: QtWidgets.QWidget = None,
#     ):
#         super().__init__(percent_min, percent_max, parent)
#         self.int_min = int_min
#         self.int_max = int_max

#     def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
#         if len(input) == 0:
#             return (QtGui.QValidator.Intermediate, input, pos)

#         # Treat as percent
#         if "%" in input:
#             return super().validate(input, pos)

#         # Treat as int
#         try:
#             i = int(input)
#         except ValueError:
#             return (QtGui.QValidator.Invalid, input, pos)

#         if self.int_min is not None and i < self.int_min:
#             return (QtGui.QValidator.Intermediate, input, pos)
#         elif self.int_max is not None and i > self.int_max:
#             return (QtGui.QValidator.Invalid, input, pos)

#         return (QtGui.QValidator.Acceptable, input, pos)


class DecimalValidatorNoZero(DecimalValidator):
    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        result = super().validate(input, pos)
        if result[0] == QtGui.QValidator.Acceptable and float(input) == 0.0:
            result = (QtGui.QValidator.Invalid, input, pos)
        return result


class DoublePrecisionDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, decimals: int, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.decimals = decimals

    def displayText(self, value: str, locale: str) -> str:
        try:
            num = float(value)
            return f"{num:.{self.decimals}f}"
        except (TypeError, ValueError):
            return str(super().displayText(value, locale))
