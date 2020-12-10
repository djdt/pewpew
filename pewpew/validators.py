from PySide2 import QtGui, QtWidgets

from typing import Callable, Tuple


class DecimalValidator(QtGui.QDoubleValidator):
    def __init__(
        self,
        bottom: float,
        top: float,
        decimals: int = 4,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(bottom, top, decimals, parent)
        self.setNotation(QtGui.QDoubleValidator.StandardNotation)

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        result, _, _ = super().validate(input, pos)
        if result == QtGui.QValidator.Intermediate:
            try:
                if float(input) > self.top():
                    return (QtGui.QValidator.Invalid, input, pos)
            except ValueError:
                pass
        return (result, input, pos)


class DecimalValidatorNoZero(DecimalValidator):
    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        result = super().validate(input, pos)
        if result[0] == QtGui.QValidator.Acceptable and float(input) == 0.0:
            result = (QtGui.QValidator.Intermediate, input, pos)
        return result


class LimitValidator(QtGui.QDoubleValidator):
    def __init__(
        self,
        bottom: float,
        top: float,
        decimals: int = 4,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(bottom, top, decimals, parent)

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        result, _, _ = super().validate(input, pos)
        if result == QtGui.QValidator.Acceptable:
            try:
                v = float(input)
                if v == self.bottom():
                    return (QtGui.QValidator.Intermediate, input, pos)
                elif v == self.top():
                    return (QtGui.QValidator.Intermediate, input, pos)
            except ValueError:  # pragma: no cover
                pass
        return (result, input, pos)


class ConditionalLimitValidator(LimitValidator):
    def __init__(
        self,
        bottom: float,
        top: float,
        decimals: int = 4,
        condition: Callable[[float], bool] = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(bottom, top, decimals, parent)
        self.condition = condition

    def setCondition(self, condition: Callable[[float], bool]) -> None:
        self.condition = condition

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        result, _, _ = super().validate(input, pos)
        if result == QtGui.QValidator.Acceptable:
            try:
                v = float(input)
                if self.condition is not None and not self.condition(v):
                    return (QtGui.QValidator.Intermediate, input, pos)
            except ValueError:  # pragma: no cover
                pass
        return (result, input, pos)


class OddIntValidator(QtGui.QIntValidator):
    def __init__(self, bottom: int, top: int, parent: QtWidgets.QWidget = None):
        super().__init__(bottom, top, parent)

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        result, _, _ = super().validate(input, pos)
        if result == QtGui.QValidator.Acceptable:
            try:
                v = float(input)
                if v % 2 == 0:
                    return (QtGui.QValidator.Intermediate, input, pos)
            except ValueError:  # pragma: no cover
                pass
        return (result, input, pos)


class PercentOrDecimalValidator(DecimalValidator):
    def __init__(
        self,
        bottom: float,
        top: float,
        decimals: int = 4,
        percent_bottom: float = 0.0,
        percent_top: float = 100.0,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(bottom, top, decimals, parent)
        self._bottom = bottom
        self._top = top
        self.percent_bottom = percent_bottom
        self.percent_top = percent_top

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        # Treat as percent
        if "%" in input:
            if not input.endswith("%") or input.count("%") > 1:
                return (QtGui.QValidator.Invalid, input, pos)
            self.setRange(self.percent_bottom, self.percent_top, self.decimals())
            return (super().validate(input.rstrip("%"), pos)[0], input, pos)
        # Treat as double
        self.setRange(self._bottom, self._top, self.decimals())
        return super().validate(input, pos)


class DoublePrecisionDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, decimals: int, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.decimals = decimals

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: int,
    ) -> QtWidgets.QWidget:  # pragma: no cover
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
    ) -> QtWidgets.QWidget:  # pragma: no cover
        lineedit = QtWidgets.QLineEdit(parent)
        lineedit.setValidator(QtGui.QDoubleValidator())
        return lineedit

    def displayText(self, value: str, locale: str) -> str:
        try:
            num = float(value)
            return f"{num:#.{self.sigfigs}g}".rstrip(".").replace(".e", "e")
        except (TypeError, ValueError):
            return str(super().displayText(value, locale))
