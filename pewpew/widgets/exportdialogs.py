from PySide2 import QtCore, QtWidgets


class CsvExportOptions(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.check_trim = QtWidgets.QCheckBox("Trim data to view.")
        self.check_trim.setChecked(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.check_trim)
        self.setLayout(layout)


class PngExportOptions(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        # self.layout_form = QtWidgets.QFormLayout()
        # self.setLayout(self.layout_form)
        self.check_trim = QtWidgets.QCheckBox("Trim data to view.")
        self.check_trim.setChecked(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.check_trim)
        self.setLayout(layout)


class VtiExportOptions(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        # self.layout_form = QtWidgets.QFormLayout()
        # self.setLayout(self.layout_form)
        self.check_trim = QtWidgets.QCheckBox("Trim data to view.")
        self.check_trim.setChecked(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.check_trim)
        self.setLayout(layout)


class ExportDialog(QtWidgets.QFileDialog):
    def __init__(self, directory: str, parent: QtWidgets.QWidget = None):
        filters = (
            "CSV files(*.csv);;Numpy archives(*.npz);;"
            "PNG images(*.png);;VTK Images(*.vti);;All files(*)"
        )
        self.options = QtWidgets.QStackedWidget()
        super().__init__(parent, "Export Laser", directory, filters)
        self.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, True)
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.setLabelText(QtWidgets.QFileDialog.FileName, "A")
        self.setViewMode(QtWidgets.QFileDialog.Detail)

        self.filterSelected.connect(self.updateOptions)

        # Grab the filename lineedit so we can hook it up
        filename_edit = self.findChild(QtWidgets.QLineEdit)
        filename_edit.textChanged.connect(self.onFileNameChanged)

        self.options_button = QtWidgets.QCheckBox("Show options...")
        self.options_button.toggled.connect(self.showOptions)

        self.csv_options = CsvExportOptions()
        self.png_options = PngExportOptions()
        self.vti_options = VtiExportOptions()
        self.options.addWidget(self.csv_options)
        self.options.addWidget(self.png_options)
        self.options.addWidget(self.vti_options)
        self.options.setVisible(False)

        # self.options_box = QtWidgets.QGroupBox("Options...")
        # self.options_box.setCheckable(True)
        # self.options_box.setChecked(False)
        # self.options_box.clicked.connect(self.showOptions)
        # options_layout = QtWidgets.QVBoxLayout()
        # options_layout.addWidget(self.options)
        # self.options_box.setLayout(options_layout)
        self.options.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed
        )

        layout = self.layout()
        layout.addWidget(self.options_button, 4, 0, 1, -1)
        layout.addWidget(self.options, 5, 0, 1, -1)

    # def sizeHint(self) -> QtCore.QSize:
    #     size = super().sizeHint()
    #     return size + self.options.sizeHint()

    def onFileNameChanged(self, name: str) -> None:
        ext = name[name.rfind("."):].lower()
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
        # self.options_button.setText("Show options..." if show else )

    def updateOptions(self, filter: str):
        self.options_button.setEnabled(True)
        if filter == "CSV files(*.csv)":
            self.options.setCurrentWidget(self.csv_options)
        elif filter == "Numpy archives(*.npz)":
            # self.options_button.setChecked(False)
            self.options_button.setChecked(False)
            self.options_button.setEnabled(False)
        elif filter == "PNG images(*.png)":
            self.options.setCurrentWidget(self.png_options)
        elif filter == "VTK Images(*.vti)":
            self.options.setCurrentWidget(self.vti_options)

    def accept(self):
        raise NotImplementedError


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    fd = ExportDialog("")
    fd.show()
    app.exec_()
