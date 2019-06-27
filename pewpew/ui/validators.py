from PyQt5 import QtGui, QtWidgets

from typing import Tuple


class DecimalValidator(QtGui.QDoubleValidator):
    def __init__(
        self, bottom: float, top: float, decimals: int, parent: QtWidgets.QWidget = None
    ):
        super().__init__(bottom, top, decimals, parent)
        self.setNotation(QtGui.QDoubleValidator.StandardNotation)

    # Make comma invalid.
    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        if "," in input:
            return (QtGui.QValidator.Invalid, input, pos)
        return super().validate(input, pos)


class DecimalValidatorNoZero(DecimalValidator):
    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        result = super().validate(input, pos)
        if result[0] == QtGui.QValidator.Acceptable and float(input) == 0.0:
            result = (QtGui.QValidator.Invalid, input, pos)
        return result


class PercentOrDecimalValidator(DecimalValidator):
    def __init__(
        self,
        bottom: float = -1e99,
        top: float = 1e99,
        decimals: int = 4,
        percent_bottom: float = 0.0,
        percent_top: float = 100.0,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(bottom, top, decimals, parent)
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


class IntListValidator(QtGui.QIntValidator):
    def __init__(
        self,
        bottom: int,
        top: int,
        delimiter: str = ",",
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(bottom, top, parent)
        self.delimiter = delimiter

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        tokens = input.split(self.delimiter)
        intermediate = False

        for token in tokens:
            result = super().validate(token, 0)[0]
            if result == QtGui.QValidator.Invalid:
                return (result, input, pos)
            elif result == QtGui.QValidator.Intermediate:
                intermediate = True

        return (
            QtGui.QValidator.Intermediate
            if intermediate
            else QtGui.QValidator.Acceptable,
            input,
            pos,
        )


class DoublePrecisionDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, decimals: int, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.decimals = decimals

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: int,
    ) -> QtWidgets.QWidget:
        lineedit = QtWidgets.QLineEdit(parent)
        lineedit.setValidator(QtGui.QDoubleValidator())
        return lineedit

    def displayText(self, value: str, locale: str) -> str:
        try:
            num = float(value)
            return f"{num:.{self.decimals}f}"
        except (TypeError, ValueError):
            return str(super().displayText(value, locale))


class DoubleSignificantFiguresDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, sigfigs: int, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.sigfigs = sigfigs

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: int,
    ) -> QtWidgets.QWidget:
        lineedit = QtWidgets.QLineEdit(parent)
        lineedit.setValidator(QtGui.QDoubleValidator())
        return lineedit

    def displayText(self, value: str, locale: str) -> str:
        try:
            num = float(value)
            return f"{num:.{self.sigfigs}g}"
        except (TypeError, ValueError):
            return str(super().displayText(value, locale))
