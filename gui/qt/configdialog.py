from PyQt5 import QtGui, QtWidgets


class ConfigForm(QtWidgets.QGroupBox):
    def __init__(self, config, title=None, parent=None):
        super().__init__(title, parent)

        self.config = config

        layout = QtWidgets.QFormLayout()
        for k, v in self.config.items():
            le = QtWidgets.QLineEdit(str(v))
            if k in ["gradient", "intercept"]:
                le.setValidator(QtGui.QDoubleValidator(-1e10, 1e10, 8))
            else:
                le.setValidator(QtGui.QDoubleValidator(0, 1e3, 4))
            layout.addRow(k.capitalize() + ":", le)
            setattr(self, k, le)

        self.setLayout(layout)


class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)

        main_layout = QtWidgets.QVBoxLayout()
        # Form layout for line edits
        self.form = ConfigForm(config, parent=self)
        # Checkbox
        self.checkAll = QtWidgets.QCheckBox("Apply configs to all images.")
        # Ok button
        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok, self)
        buttonBox.accepted.connect(self.accept)

        main_layout.addWidget(self.form)
        main_layout.addWidget(self.checkAll)
        main_layout.addWidget(buttonBox)
        self.setLayout(main_layout)

        self.resize(540, 320)

    def accept(self):
        for k in self.form.config.keys():
            v = float(getattr(self.form, k).text())
            self.form.config[k] = v
        super().accept()
