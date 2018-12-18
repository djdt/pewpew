from PyQt5 import QtGui


class PercentValidator(QtGui.QValidator):
    def __init__(self, min_value=0, max_value=100, parent=None):
        super().__init__(parent)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, input, pos):
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


