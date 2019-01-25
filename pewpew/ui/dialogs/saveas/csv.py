from PyQt5 import QtWidgets

from .saveas import SaveAsDialog

from pewpew.lib.io import csv

from pewpew.lib.laser import LaserData


class CSVSaveAsDialog(SaveAsDialog):
    def __init__(
        self,
        path: str,
        isotope: str,
        isotopes: int = -1,
        layers: int = -1,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(path, isotope, isotopes, layers, parent)

        self.check_trim = QtWidgets.QCheckBox("Trim data.")
        self.check_trim.setChecked()
        self.check_header = QtWidgets.QCheckBox("Include header.")
        self.check_header.setChecked()

        options_layout = QtWidgets.QVBoxLayout()
        options_layout.addWidget(self.check_trim)
        options_layout.addWidget(self.check_header)

        self.options_box.setLayout(options_layout)

    def _save(self, path: str, isotope: str, layer: int, laser: LaserData) -> None:
        csv.save(
            path,
            laser,
            isotope,
            trimmed=self.check_trim.isChecked(),
            include_header=self.check_header.isCheck(),
        )
