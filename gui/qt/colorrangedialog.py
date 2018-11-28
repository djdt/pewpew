from PyQt5 import QtWidgets, QtGui


class IntOrPercentValidator(QtGui.QIntValidator):
    def __init__(self, min_int=None, max_int=None, parent=None):
        super().__init__(parent)
        self.min_int = min_int
        self.max_int = max_int

    def validate(self, input, pos):
        if len(input) == 0:
            return (QtGui.QValidator.Intermediate, input, pos)

        if input.endswith('%'):
            if input.count('%') > 1:
                return (QtGui.QValidator.Invalid, input, pos)
            min_int = 0
            max_int = 100
        else:
            min_int = self.min_int
            max_int = self.max_int

        try:
            i = int(input.rstrip('%'))
        except ValueError:
            return (QtGui.QValidator.Invalid, input, pos)

        if min_int is not None and i < min_int:
            return (QtGui.QValidator.Intermediate, input, pos)

        if max_int is not None and i > max_int:
            return (QtGui.QValidator.Invalid, input, pos)

        return (QtGui.QValidator.Acceptable, input, pos)


class ColorRangeDialog(QtWidgets.QDialog):
    def __init__(self, current_range, parent=None):
        self.range = current_range
        super().__init__(parent)
        self.setWindowTitle("Colormap Range")

        self.lineedit_min = QtWidgets.QLineEdit()
        self.lineedit_min.setPlaceholderText(str(current_range[0]))
        self.lineedit_min.setToolTip("Enter absolute value or percentile.")
        self.lineedit_min.setValidator(IntOrPercentValidator(
            min_int=0, parent=self))
        self.lineedit_max = QtWidgets.QLineEdit()
        self.lineedit_max.setPlaceholderText(str(current_range[1]))
        self.lineedit_min.setValidator(IntOrPercentValidator(
            min_int=0, parent=self))
        self.lineedit_max.setToolTip("Enter absolute value or percentile.")

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Minimum:", self.lineedit_min)
        form_layout.addRow("Maximum:", self.lineedit_max)

        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        main_layout = QtWidgets.QVBoxLayout()

        main_layout.addLayout(form_layout)
        main_layout.addWidget(buttonBox)
        self.setLayout(main_layout)

    def getRangeAsFloatOrPercent(self):
        minimum = self.lineedit_min.text()
        if len(minimum) == 0:
            minimum = self.range[0]
        elif not minimum.endswith('%'):
            minimum = int(minimum)
        maximum = self.lineedit_max.text()
        if len(maximum) == 0:
            maximum = self.range[1]
        elif not maximum.endswith('%'):
            maximum = int(maximum)

        return (minimum, maximum)
