from PyQt5 import QtWidgets

from .export import ExportDialog


class CSVExportOptions(QtWidgets.QGroupBox):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("CSV Options", parent)
        self.check_trim = QtWidgets.QCheckBox("Trim data to view.")
        self.check_trim.setChecked(True)
        # self.check_header = QtWidgets.QCheckBox("Include header.")
        # self.check_header.setChecked(False)

        options_layout = QtWidgets.QVBoxLayout()
        options_layout.addWidget(self.check_trim)
        # options_layout.addWidget(self.check_header)

        self.setLayout(options_layout)

    # def headerChecked(self) -> bool:
    #     return self.check_header.isChecked()

    def trimmedChecked(self) -> bool:
        return self.check_trim.isChecked()


class CSVExportDialog(ExportDialog):
    def __init__(
        self,
        path: str,
        name: str,
        names: int = -1,
        layers: int = -1,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(path, name, names, layers, CSVExportOptions(), parent)
