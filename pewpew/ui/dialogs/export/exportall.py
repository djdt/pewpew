import os.path

from PyQt5 import QtWidgets

from pewpew.ui.dialogs.export.export import ExportDialog
from pewpew.ui.dialogs.export.csv import CSVExportOptions
from pewpew.ui.dialogs.export.png import PNGExportOptions

from typing import List
from pewpew.lib.laser import LaserData


class ExportAllOptions(QtWidgets.QStackedWidget):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.default = QtWidgets.QWidget()
        self.csv = CSVExportOptions()
        self.png = PNGExportOptions()

        self.addWidget(self.default)
        self.addWidget(self.csv)
        self.addWidget(self.png)

    def update(self, ext: str) -> None:
        if ext == ".csv":
            self.setCurrentWidget(self.csv)
        elif ext == ".png":
            self.setCurrentWidget(self.png)
        else:
            self.setCurrentWidget(self.default)


class ExportAllDialog(ExportDialog):
    FORMATS = {
        "CSV File": ".csv",
        "Numpy Archive": ".npz",
        "PNG Image": ".png",
        "VTK Image": ".vti",
    }

    def __init__(
        self,
        path: str,
        name: str,
        isotopes: List[str],
        layers: int,
        parent: QtWidgets.QWidget = None,
    ):
        self.name = name
        layout_all = QtWidgets.QFormLayout()
        self.lineedit_prefix = QtWidgets.QLineEdit()
        self.lineedit_prefix.textChanged.connect(self._update)

        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.addItems(isotopes)
        self.combo_isotopes.currentIndexChanged.connect(self._update)

        self.combo_formats = QtWidgets.QComboBox()
        self.combo_formats.addItems(ExportAllDialog.FORMATS.keys())
        self.combo_formats.currentIndexChanged.connect(self._update)

        super().__init__(
            path, isotopes[0], len(isotopes), layers, ExportAllOptions(), parent
        )

        layout_all.addRow("Prefix:", self.lineedit_prefix)
        layout_all.addRow("Isotope:", self.combo_isotopes)
        layout_all.addRow("Format:", self.combo_formats)

        self.layout().insertLayout(0, layout_all)

    def _update(self) -> None:
        ext = ExportAllDialog.FORMATS[self.combo_formats.currentText()]
        self.options.update(ext)
        if ext in [".npz", ".vti"]:
            self.check_isotopes.setEnabled(False)
            self.check_isotopes.setChecked(True)
        elif not self.check_isotopes.isEnabled():
            self.check_isotopes.setEnabled(True)
            self.check_isotopes.setChecked(False)
        self.combo_isotopes.setEnabled(not self.check_isotopes.isChecked())

        self.lineedit_preview.setText(
            os.path.basename(
                self._generate_path(
                    isotope=self.combo_isotopes.currentText()
                    if self.check_isotopes.isChecked()
                    and self.check_isotopes.isEnabled()
                    else None
                )
            )
        )

    def _get_isotope(self) -> str:
        return self.combo_isotopes.currentText()

    def _generate_path(
        self, laser: LaserData = None, isotope: str = None, layer: int = None
    ) -> str:
        if laser is not None and isotope is not None:
            if isotope not in laser.isotopes():
                return ""
        name = self.name if laser is None else laser.name

        prefix = self.lineedit_prefix.text()
        path = os.path.join(self.path, prefix + "_" + name if prefix != "" else name)
        ext = ExportAllDialog.FORMATS[self.combo_formats.currentText()]
        if isotope is not None:
            path += f"_{isotope}"
        if layer is not None:
            path += f"_{layer}"
        return path + ext
