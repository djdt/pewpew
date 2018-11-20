from PyQt5 import QtCore, QtWidgets

from util.laser import LaserData


class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, config=LaserData.DEFAULT_CONFIG):
        super().__init__(parent)

        self.config = config

        mainLayout = QtWidgets.QVBoxLayout()
        # Form layout for line edits
        form = QtWidgets.QGroupBox()
        formLayout = QtWidgets.QFormLayout()
        for k, v in self.config.items():
            le = QtWidgets.QLineEdit(str(v))
            formLayout.addRow(k.capitalize() + ":", le)
            setattr(self, k + "LineEdit", le)
        form.setLayout(formLayout)
        # Checkbox
        self.checkAll = QtWidgets.QCheckBox("Apply configs to all images.")
        # Ok button
        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok, self)
        buttonBox.accepted.connect(self.accept)

        mainLayout.addWidget(form)
        mainLayout.addWidget(self.checkAll)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

        self.resize(540, 320)

    def accept(self):
        for k in self.config.keys():
            v = float(getattr(self, k + "LineEdit").text())
            self.config[k] = v
        super().accept()
