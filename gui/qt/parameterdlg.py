from PyQt5 import QtCore, QtWidgets

from util.laser import LaserParams


class ParameterDialog(QtWidgets.QDialog):
    EDITABLE_PARAMS = ['spotsize', 'speed', 'scantime', 'gradient',
                       'intercept']

    def __init__(self, parent=None, parameters=LaserParams()):
        super().__init__(parent)

        self.resize(540, 320)

        mainLayout = QtWidgets.QVBoxLayout()

        form = QtWidgets.QGroupBox()
        formLayout = QtWidgets.QFormLayout()
        for p in ParameterDialog.EDITABLE_PARAMS:
            le = QtWidgets.QLineEdit(str(getattr(parameters, p, 0.0)))
            formLayout.addRow(p.capitalize() + ":", le)
            setattr(self, p + "LineEdit", le)
        form.setLayout(formLayout)

        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.addButton("&Update",
                                 QtWidgets.QDialogButtonBox.AcceptRole)
        self.buttonBox.addButton("Update &All",
                                 QtWidgets.QDialogButtonBox.AcceptRole)
        # self.buttonBox.addButton("&Close",
        #                          QtWidgets.QDialogButtonBox.RejectRole)
        mainLayout.addWidget(form)
        mainLayout.addWidget(self.buttonBox)
        self.setLayout(mainLayout)
