from PySide2 import QtCore, QtGui, QtWidgets

from laserlib.laser import Laser

from typing import Tuple


class CsvExportOptions(QtWidgets.QGroupBox):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

        self.check_trim = QtWidgets.QCheckBox("Trim data to view.")
        self.check_trim.setChecked(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.check_trim)
        self.setLayout(layout)


class PngExportOptions(QtWidgets.QGroupBox):
    def __init__(self, imagesize: Tuple[int, int], parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

        self.linedit_size_x = QtWidgets.QLineEdit(str(imagesize[0]))
        self.linedit_size_x.setValidator(QtGui.QIntValidator(0, 9999))
        self.linedit_size_y = QtWidgets.QLineEdit(str(imagesize[1]))
        self.linedit_size_y.setValidator(QtGui.QIntValidator(0, 9999))

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Size:"), 0)
        layout.addWidget(self.linedit_size_x, 0)
        layout.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.linedit_size_y, 0)
        layout.addStretch(1)

        self.setLayout(layout)


class VtiExportOptions(QtWidgets.QGroupBox):
    def __init__(
        self, spacing: Tuple[float, float, float], parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        x, y, z = spacing

        self.linedit_size_x = QtWidgets.QLineEdit(str(x))
        self.linedit_size_x.setEnabled(False)
        # self.linedit_size_x.setValidator(DecimalValidator(0, 1e99, 4))
        self.linedit_size_y = QtWidgets.QLineEdit(str(y))
        self.linedit_size_y.setEnabled(False)
        # self.linedit_size_y.setValidator(DecimalValidator(0, 1e99, 4))
        self.linedit_size_z = QtWidgets.QLineEdit(str(z))
        self.linedit_size_z.setValidator(QtGui.QDoubleValidator(-1e99, 1e99, 4))

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Spacing:"), 0)
        layout.addWidget(self.linedit_size_x, 0)
        layout.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.linedit_size_y, 0)
        layout.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.linedit_size_z, 0)
        layout.addStretch(1)

        self.setLayout(layout)


class ExportDialog(QtWidgets.QFileDialog):
    def __init__(self, laser: Laser, directory: str, parent: QtWidgets.QWidget = None):
        super().__init__(
            parent,
            "Export Laser",
            directory,
            "CSV files(*.csv);;Numpy archives(*.npz);;"
            "PNG images(*.png);;VTK Images(*.vti);;All files(*)",
        )
        self.setStyle(QtWidgets.QMacStyle())
        self.resize(QtCore.QSize(800, 600))
        self.laser = laser

        self.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, True)
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.setViewMode(QtWidgets.QFileDialog.Detail)

        self.filterSelected.connect(self.updateOptions)

        # Grab the filename lineedit so we can hook it up
        filename_edit = self.findChild(QtWidgets.QLineEdit)
        filename_edit.textChanged.connect(self.onFileNameChanged)

        self.check_all_isotopes = QtWidgets.QCheckBox("Export all isotopes.")
        self.check_all_isotopes.setChecked(False)

        self.options_button = QtWidgets.QCheckBox("Format options...")
        self.options_button.toggled.connect(self.showOptions)

        self.csv_options = CsvExportOptions()
        self.png_options = PngExportOptions((1280, 800))
        spacing = (
            self.laser.config.get_pixel_width(),
            self.laser.config.get_pixel_height(),
            self.laser.config.spotsize / 2.0,
        )
        self.vti_options = VtiExportOptions(spacing)

        self.options = QtWidgets.QStackedWidget()
        self.options.addWidget(self.csv_options)
        self.options.addWidget(self.png_options)
        self.options.addWidget(self.vti_options)
        self.options.setVisible(False)
        self.options.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )

        layout = self.layout()
        layout.addWidget(self.check_all_isotopes, 4, 0, 1, -1)
        layout.addWidget(self.options_button, 5, 0, 1, -1)
        layout.addWidget(self.options, 6, 0, 1, -1)

    def onFileNameChanged(self, name: str) -> None:
        ext = name[name.rfind(".") :].lower()
        if ext == ".csv":
            self.selectNameFilter("CSV files(*.csv)")
        elif ext == ".npz":
            self.selectNameFilter("Numpy archives(*.npz)")
        elif ext == ".png":
            self.selectNameFilter("PNG images(*.png)")
        elif ext == ".vti":
            self.selectNameFilter("VTK Images(*.vti)")
        else:
            return
        self.updateOptions(self.selectedNameFilter())

    def showOptions(self, show: bool) -> None:
        self.options.setVisible(show)

    def updateOptions(self, filter: str):
        if filter == "CSV files(*.csv)":
            self.options_button.setEnabled(True)
            self.options.setCurrentWidget(self.csv_options)
        elif filter == "Numpy archives(*.npz)":
            self.options_button.setEnabled(False)
        elif filter == "PNG images(*.png)":
            self.options_button.setEnabled(True)
            self.options.setCurrentWidget(self.png_options)
        elif filter == "VTK Images(*.vti)":
            self.options_button.setEnabled(True)
            self.options.setCurrentWidget(self.vti_options)

        if self.options_button.isChecked() and self.options_button.isEnabled():
            self.showOptions(True)
        else:
            self.showOptions(False)

    def accept(self):
        path = self.selectedFiles()[0]
        ext = self.selectedNameFilter()
        ext = ext[ext.rfind(".") : -1]
        print(path)
        super().accept()

    # print('what')
    # raise NotImplementedError


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    fd = ExportDialog(Laser(), "")
    fd.show()
    app.exec_()
