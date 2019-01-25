from PyQt5 import QtGui, QtWidgets

from .saveas import SaveAsDialog

from pewpew.lib.io import png

from typing import Union
from pewpew.lib.laser import LaserData


class PNGSaveAsDialog(SaveAsDialog):
    def __init__(
        self,
        path: str,
        isotope: str,
        viewconfig: dict,
        isotopes: int = -1,
        layers: int = -1,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(path, isotope, isotopes, layers, parent)
        self.viewconfig = viewconfig

        self.linedit_size_x = QtWidgets.QLineEdit("640")
        self.linedit_size_x.setValidator(QtGui.QIntValidator(0, 9999))
        self.linedit_size_y = QtWidgets.QLineEdit("480")
        self.linedit_size_y.setValidator(QtGui.QIntValidator(0, 9999))

        self.check_colorbar = QtWidgets.QCheckBox("Include color bar.")
        self.check_scalebar = QtWidgets.QCheckBox("Include scale bar.")
        self.check_label = QtWidgets.QCheckBox("Include isotope label.")

        layout_size = QtWidgets.QHBoxLayout()
        layout_size.addWidget(QtWidgets.QLabel("Size:"))
        layout_size.addWidget(self.linedit_size_x)
        layout_size.addWidget(self.linedit_size_y)

        options_layout = QtWidgets.QVBoxLayout()
        options_layout.addLayout(layout_size)
        options_layout.addWidget(self.check_colorbar)
        options_layout.addWidget(self.check_scalebar)
        options_layout.addWidget(self.check_label)

        self.options_box.setLayout(options_layout)

    def _save(self, path: str, isotope: str, layer: int, laser: LaserData) -> None:

        size = (int(self.linedit_size_x.text()), int(self.linedit_size_y.text()))
        png.save(
            path,
            laser,
            isotope,
            self.viewconfig,
            size=size,
            include_colorbar=self.check_colorbar.isChecked(),
            include_scalebar=self.check_scalebar.isChecked(),
            include_label=self.check_label.isChecked(),
        )
