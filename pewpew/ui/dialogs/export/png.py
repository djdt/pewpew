from PyQt5 import QtGui, QtWidgets

from .export import ExportDialog

from typing import Tuple


class PNGExportOptions(QtWidgets.QGroupBox):
    def __init__(
        self, imagesize: Tuple[int, int] = (1280, 800), parent: QtWidgets.QWidget = None
    ):
        super().__init__("PNG Options", parent)

        self.linedit_size_x = QtWidgets.QLineEdit(str(imagesize[0]))
        self.linedit_size_x.setValidator(QtGui.QIntValidator(0, 9999))
        self.linedit_size_y = QtWidgets.QLineEdit(str(imagesize[1]))
        self.linedit_size_y.setValidator(QtGui.QIntValidator(0, 9999))

        # We can just use the open options
        # self.check_colorbar = QtWidgets.QCheckBox("Include color bar.")
        # self.check_colorbar.setChecked(True)
        # self.check_scalebar = QtWidgets.QCheckBox("Include scale bar.")
        # self.check_scalebar.setChecked(True)
        # self.check_label = QtWidgets.QCheckBox("Include isotope label.")
        # self.check_label.setChecked(True)

        layout_size = QtWidgets.QHBoxLayout()
        layout_size.addWidget(QtWidgets.QLabel("Size:"))
        layout_size.addWidget(self.linedit_size_x)
        layout_size.addWidget(self.linedit_size_y)

        options_layout = QtWidgets.QVBoxLayout()

        options_layout.addLayout(layout_size)
        # options_layout.addWidget(self.check_colorbar)
        # options_layout.addWidget(self.check_scalebar)
        # options_layout.addWidget(self.check_label)

        self.setLayout(options_layout)

    def imagesize(self) -> Tuple[int, int]:
        return (int(self.linedit_size_x.text()), int(self.linedit_size_y.text()))


#     def colorbarChecked(self) -> bool:
#         return self.check_colorbar.isChecked()

#     def scalebarChecked(self) -> bool:
#         return self.check_scalebar.isChecked()

#     def labelChecked(self) -> bool:
#         return self.check_label.isChecked()


class PNGExportDialog(ExportDialog):
    def __init__(
        self,
        path: str,
        name: str,
        names: int = -1,
        layers: int = -1,
        viewlimits: Tuple[float, float, float, float] = None,
        parent: QtWidgets.QWidget = None,
    ):
        imagesize = (1280, 800)
        if viewlimits is not None:
            x = viewlimits[1] - viewlimits[0]
            y = viewlimits[3] - viewlimits[2]
            imagesize = (1280, int(1280 * x / y)) if x > y else (int(800 * y / x), 800)
        super().__init__(
            path, name, names, layers, PNGExportOptions(imagesize=imagesize), parent
        )
