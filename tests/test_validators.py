from PySide2 import QtCore, QtGui

from pewpew import validators

QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.English))


def test_decimal_validator():
    validator = validators.DecimalValidator(10.0, 100.0, 3)
    # Exponent
    assert validator.validate("1e", 0)[0] == QtGui.QValidator.Invalid
    # Text
    assert validator.validate("a", 0)[0] == QtGui.QValidator.Invalid
    assert validator.validate(" ", 0)[0] == QtGui.QValidator.Invalid
    # Intermediate
    assert validator.validate("1,", 0)[0] == QtGui.QValidator.Intermediate
    assert validator.validate("9", 0)[0] == QtGui.QValidator.Intermediate
    # Range
    assert validator.validate("-1", 0)[0] == QtGui.QValidator.Invalid
    assert validator.validate("101.", 4)[0] == QtGui.QValidator.Invalid
    # Decimals
    assert validator.validate("11.0", 0)[0] == QtGui.QValidator.Acceptable
    assert validator.validate("11.01", 0)[0] == QtGui.QValidator.Acceptable
    assert validator.validate("11.001", 0)[0] == QtGui.QValidator.Acceptable
    assert validator.validate("11.0001", 0)[0] == QtGui.QValidator.Invalid


def test_decimal_validator_no_zero():
    validator = validators.DecimalValidatorNoZero(-100.0, 100.0, 3)
    # Reject zero
    assert validator.validate("0.", 0)[0] == QtGui.QValidator.Intermediate
    assert validator.validate("-0", 0)[0] == QtGui.QValidator.Intermediate


def test_limit_validator():
    validator = validators.LimitValidator(0.0, 1.0, 3)
    assert validator.validate("0.1", 0)[0] == QtGui.QValidator.Acceptable
    assert validator.validate("0.", 0)[0] == QtGui.QValidator.Intermediate
    assert validator.validate("1.", 0)[0] == QtGui.QValidator.Intermediate
    assert validator.validate("1.1", 0)[0] == QtGui.QValidator.Intermediate


def test_odd_int_validator():
    validator = validators.OddIntValidator(0, 10)
    assert validator.validate("2", 0)[0] == QtGui.QValidator.Intermediate
    assert validator.validate("1", 0)[0] == QtGui.QValidator.Acceptable
    assert validator.validate("11", 0)[0] == QtGui.QValidator.Intermediate


def test_percent_or_decimal_validator():
    validator = validators.PercentOrDecimalValidator(-100.0, 100.0, 3, 2, 10)
    # Accept one percent sign
    assert validator.validate("%", 0)[0] == QtGui.QValidator.Intermediate
    assert validator.validate("%%", 0)[0] == QtGui.QValidator.Invalid
    assert validator.validate("2%", 0)[0] == QtGui.QValidator.Acceptable
    assert validator.validate("%2", 0)[0] == QtGui.QValidator.Invalid
    # Percent range
    assert validator.validate("1%", 0)[0] == QtGui.QValidator.Intermediate
    assert validator.validate("2%", 0)[0] == QtGui.QValidator.Acceptable
    assert validator.validate("11%", 0)[0] == QtGui.QValidator.Invalid
    # Decimal mode
    assert validator.validate("2.0", 0)[0] == QtGui.QValidator.Acceptable
    assert validator.validate("111", 0)[0] == QtGui.QValidator.Invalid


def test_double_precision_delegate():
    delegate = validators.DoublePrecisionDelegate(4)
    # General
    assert delegate.displayText("1", "en") == "1.0000"
    assert delegate.displayText("1.11e1", "en") == "11.1000"
    # Rounding
    assert delegate.displayText("1.00001", "en") == "1.0000"
    assert delegate.displayText("1.00009", "en") == "1.0001"
    # Passes text
    assert delegate.displayText("a", "en") == "a"


def test_double_significant_figures_delegate():
    delegate = validators.DoubleSignificantFiguresDelegate(4)
    # General
    assert delegate.displayText("1", "en") == "1.000"
    assert delegate.displayText("10", "en") == "10.00"
    assert delegate.displayText("100", "en") == "100.0"
    assert delegate.displayText("1000", "en") == "1000"
    assert delegate.displayText("1.1e3", "en") == "1100"
    # Exponent and rounding
    assert delegate.displayText("10101", "en") == "1.010e+04"
    assert delegate.displayText("10109", "en") == "1.011e+04"
    # Passes text
    assert delegate.displayText("a", "en") == "a"
