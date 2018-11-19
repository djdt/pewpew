from PyQt5 import QtCore, QtWidgets

from util.laser import LaserConfig


class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, configs=LaserConfig()):
        super().__init__(parent)

        self.resize(540, 320)

        mainLayout = QtWidgets.QVBoxLayout()

        form = QtWidgets.QGroupBox()
        formLayout = QtWidgets.QFormLayout()
        for p in LaserConfig.EDITABLE:
            le = QtWidgets.QLineEdit(str(getattr(configs, p, 0.0)))
            formLayout.addRow(p.capitalize() + ":", le)
            setattr(self, p + "LineEdit", le)
        form.setLayout(formLayout)

        self.checkAll = QtWidgets.QCheckBox("Apply configs to all images.")

        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok, self)
        buttonBox.accepted.connect(self.accept)

        mainLayout.addWidget(form)
        mainLayout.addWidget(self.checkAll)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

    def configs(self):
        config = LaserConfig()
        for p in LaserConfig.EDITABLE:
            v = float(getattr(self, p + "LineEdit").text())
            setattr(config, p, v)
        return config
