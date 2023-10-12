from PySide6 import QtCore, QtGui, QtWidgets

from typing import Callable,  Tuple


class DecimalValidator(QtGui.QDoubleValidator):
    """Double validator that forbids scientific notation."""

    def __init__(
        self,
        bottom: float,
        top: float,
        decimals: int = 4,
        parent: QtWidgets.QWidget | None = None,
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
    """DecimalValidator that also forbids zeros."""

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        result = super().validate(input, pos)
        if result[0] == QtGui.QValidator.Acceptable and float(input) == 0.0:
            result = (QtGui.QValidator.Intermediate, input, pos)
        return result


class LimitValidator(QtGui.QDoubleValidator):
    """QDoubleValidator that forbids top and bottom values.

    A normal QDoubleValidator allows [bottom, top] while LimitValidator allows (bottom, top).
    """

    def __init__(
        self,
        bottom: float,
        top: float,
        decimals: int = 4,
        parent: QtWidgets.QWidget | None = None,
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
    """QDoubleValidator that requires a condition to be filled.

    e.g. to forbit even numbers pass 'lambda x: x % 2 != 0'.
    If condition is None functions as a normal QDoubleValidator.
    """

    def __init__(
        self,
        bottom: float,
        top: float,
        decimals: int = 4,
        condition: Callable[[float], bool] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(bottom, top, decimals, parent)
        self.condition = condition

    def setCondition(self, condition: Callable[[float], bool]) -> None:
        """Set the valid condition."""
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
    """QIntValidator that only accepts odd numbers."""

    def __init__(self, bottom: int, top: int, parent: QtWidgets.QWidget | None = None):
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
    """DecimalValidator that accepts inputs as a percent.

    Inputs that end with '%' are treated as a percentage input.

    Args:
        bottom: decimal lower bound
        top: decimal upper bound
        decimals: number of decimals allowed
        percent_bottom: percent lower bound
        percent_top: percent upper bound
        parent: parent widget
    """

    def __init__(
        self,
        bottom: float,
        top: float,
        decimals: int = 4,
        percent_bottom: float = 0.0,
        percent_top: float = 100.0,
        parent: QtWidgets.QWidget | None = None,
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
    """Delegate to display items with a certain number of decimals.

    Item inputs are also validated using a QDoubleValidator.
    """

    def __init__(self, decimals: int, parent: QtCore.QObject | None = None):
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

    def displayText(self, value: str, locale: QtCore.QLocale) -> str:
        """Attempts to display text as a float with 'decimal' places."""
        try:
            num = float(value)
            return f"{num:.{self.decimals}f}"
        except (TypeError, ValueError):
            return str(super().displayText(value, locale))


class DoubleSignificantFiguresDelegate(QtWidgets.QStyledItemDelegate):
    """Delegate to display items with a certain number of significant figures.

    Item inputs are also validated using a QDoubleValidator.
    """
    def __init__(self, sigfigs: int, parent: QtCore.QObject | None = None):
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

    def displayText(self, value: str, locale: QtCore.QLocale) -> str:
        """Attempts to display text as a float with 'sigfigs' places."""
        try:
            num = float(value)
            return f"{num:#.{self.sigfigs}g}".rstrip(".").replace(".e", "e")
        except (TypeError, ValueError):
            return str(super().displayText(value, locale))
