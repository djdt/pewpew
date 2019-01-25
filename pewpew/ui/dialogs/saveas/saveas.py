import os.path

from PyQt5 import QtWidgets

from pewpew.lib.laser import LaserData


class SaveAsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        path: str,
        isotope: str,
        isotopes: int,
        layers: int,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.path = path
        self.isotope = isotope

        self.setWindowTitle("SaveAs")

        self.options_box = QtWidgets.QGroupBox("Options")

        self.check_isotopes = QtWidgets.QCheckBox("Save all isotopes.")
        if isotopes < 2:
            self.check_isotopes.setEnabled(False)
        self.check_isotopes.stateChanged(self.drawPreview())
        # self.check_layers = QtWidgets.QCheckBox("Save all layers.")
        # if layers < 2:
        #     self.check_layers.setEnabled(False)
        # self.check_layers.stateChanged(self.drawPreview())

        self.lineedit_preview = QtWidgets.QLineEdit()
        self.lineedit_preview.setEnabled(False)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.layoutWidgets()

        self.drawPreview()

    def layoutWidgets(self) -> None:
        layout_preview = QtWidgets.QHBoxLayout()
        layout_preview.addWidget(QtWidgets.QLabel("Preview:"))
        layout_preview.addWidget(self.lineedit_preview)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addWidget(self.options_box)
        layout_main.addWidget(self.check_isotopes)
        layout_main.addWidget(self.check_layers)
        layout_main.addLayout(layout_preview)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

    def drawPreview(self):
        name, ext = os.path.splitext(os.path.basename(self.path))
        if self.check_isotopes.isChecked():
            name += self.isotope
        # if self.check_layers.isChecked():
        #     name += "_1"
        self.lineedit_preview.setText(name + ext)

    def _save(self, path: str, isotope: str, layer: int, laser: LaserData):
        pass

    def saveAs(self, laser: LaserData):
        isotopes = laser.isotopes() if self.check_isotopes.isChecked() else [None]
        # layers = laser.layers() if self.check_layers.isChecked() else [None]
        layers = [None]
        for isotope in isotopes:
            for layer in layers:
                name, ext = os.path.splitext(self.path)
                if isotope is not None:
                    name += f"_{isotope}"
                if layer is not None:
                    name += f"_{layer}"
            self._save(name + ext, isotope, layer, laser)
