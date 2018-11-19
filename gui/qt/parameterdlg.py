from PyQt5 import QtCore, QtWidgets

from util.laser import LaserParams


class ParameterDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, parameters=LaserParams()):
        super().__init__(parent)

        self.resize(540, 320)

        mainLayout = QtWidgets.QVBoxLayout()

        form = QtWidgets.QGroupBox()
        formLayout = QtWidgets.QFormLayout()
        for p in LaserParams.EDITABLE_PARAMS:
            le = QtWidgets.QLineEdit(str(getattr(parameters, p, 0.0)))
            formLayout.addRow(p.capitalize() + ":", le)
            setattr(self, p + "LineEdit", le)
        form.setLayout(formLayout)

        self.checkAll = QtWidgets.QCheckBox("Apply parameters to all images.")

        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok, self)
        buttonBox.accepted.connect(self.accept)

        mainLayout.addWidget(form)
        mainLayout.addWidget(self.checkAll)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

    def parameters(self):
        params = LaserParams()
        for p in LaserParams.EDITABLE_PARAMS:
            v = float(getattr(self, p + "LineEdit").text())
            setattr(params, p, v)
        return params
