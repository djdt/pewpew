from PyQt5 import QtGui, QtWidgets

from .export import ExportDialog

from typing import Tuple


class PNGExportOptions(QtWidgets.QGroupBox):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("PNG Options", parent)

        self.linedit_size_x = QtWidgets.QLineEdit("1280")
        self.linedit_size_x.setValidator(QtGui.QIntValidator(0, 9999))
        self.linedit_size_y = QtWidgets.QLineEdit("800")
        self.linedit_size_y.setValidator(QtGui.QIntValidator(0, 9999))

        self.check_colorbar = QtWidgets.QCheckBox("Include color bar.")
        self.check_colorbar.setChecked(True)
        self.check_scalebar = QtWidgets.QCheckBox("Include scale bar.")
        self.check_scalebar.setChecked(True)
        self.check_label = QtWidgets.QCheckBox("Include isotope label.")
        self.check_label.setChecked(True)

        layout_size = QtWidgets.QHBoxLayout()
        layout_size.addWidget(QtWidgets.QLabel("Size:"))
        layout_size.addWidget(self.linedit_size_x)
        layout_size.addWidget(self.linedit_size_y)

        options_layout = QtWidgets.QVBoxLayout()

        options_layout.addLayout(layout_size)
        options_layout.addWidget(self.check_colorbar)
        options_layout.addWidget(self.check_scalebar)
        options_layout.addWidget(self.check_label)

        self.setLayout(options_layout)

    def imagesize(self) -> Tuple[int, int]:
        return (int(self.linedit_size_x.text()), int(self.linedit_size_y.text()))

    def colorbarChecked(self) -> bool:
        return self.check_colorbar.isChecked()

    def scalebarChecked(self) -> bool:
        return self.check_scalebar.isChecked()

    def labelChecked(self) -> bool:
        return self.check_label.isChecked()


class PNGExportDialog(ExportDialog):
    def __init__(
        self,
        path: str,
        isotope: str,
        isotopes: int = -1,
        layers: int = -1,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(path, isotope, isotopes, layers, PNGExportOptions(), parent)
