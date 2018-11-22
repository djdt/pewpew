from PyQt5 import QtCore, QtWidgets

from util.laser import LaserData


class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, config=LaserData.DEFAULT_CONFIG):
        super().__init__(parent)

        self.config = config

        main_layout = QtWidgets.QVBoxLayout()
        # Form layout for line edits
        form = QtWidgets.QGroupBox()
        form_layout = QtWidgets.QFormLayout()
        for k, v in self.config.items():
            le = QtWidgets.QLineEdit(str(v))
            form_layout.addRow(k.capitalize() + ":", le)
            setattr(self, k + "LineEdit", le)
        form.setLayout(form_layout)
        # Checkbox
        self.checkAll = QtWidgets.QCheckBox("Apply configs to all images.")
        # Ok button
        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok, self)
        buttonBox.accepted.connect(self.accept)

        main_layout.addWidget(form)
        main_layout.addWidget(self.checkAll)
        main_layout.addWidget(buttonBox)
        self.setLayout(main_layout)

        self.resize(540, 320)

    def accept(self):
        for k in self.config.keys():
            v = float(getattr(self, k + "LineEdit").text())
            self.config[k] = v
        super().accept()
