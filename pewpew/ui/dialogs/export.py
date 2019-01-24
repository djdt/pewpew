import os.path

from PyQt5 import QtWidgets

from typing import List
from pewpew.lib.laser import LaserData


class ExportDialog(QtWidgets.QDialog):
    VALID_FORMATS = [".csv", ".npz", ".vtr"]

    def __init__(
        self,
        lasers: List[LaserData],
        default_path: str = "",
        default_isotope: str = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export Data")
        self.names = [laser.name for laser in lasers]
        self.isotopes: List[str] = []
        for laser in lasers:
            self.isotopes.extend(i for i in laser.isotopes() if i not in self.isotopes)
        self.default_isotope = default_isotope

        self.layers = max(laser.layers() for laser in lasers)

        self.lineedit_path = QtWidgets.QLineEdit(default_path)
        self.button_path = QtWidgets.QPushButton("Select")

        self.check_isotopes = QtWidgets.QCheckBox("Export all isotopes.")
        self.check_layers = QtWidgets.QCheckBox("Export all layers.")

        self.combo_isotopes = QtWidgets.QComboBox()

        self.lineedit_preview = QtWidgets.QLineEdit()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )

        self.initialiseWidgets()
        self.layoutWidgets()

        self.contextEnable()
        self.drawPreview()

    def initialiseWidgets(self) -> None:
        self.lineedit_path.setMinimumWidth(300)
        self.lineedit_path.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.lineedit_path.textEdited.connect(self.changed)
        self.lineedit_path.setSelection(
            len(os.path.dirname(self.lineedit_path.text())) + 1,
            len(os.path.basename(self.lineedit_path.text())),
        )

        self.button_path.pressed.connect(self.buttonPath)

        self.check_isotopes.stateChanged.connect(self.changed)
        self.check_layers.stateChanged.connect(self.changed)

        self.combo_isotopes.addItems(self.isotopes)
        if self.default_isotope is not None:
            self.combo_isotopes.setCurrentText(self.default_isotope)
        self.combo_isotopes.currentIndexChanged.connect(self.changed)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.lineedit_preview = QtWidgets.QLineEdit()
        self.lineedit_preview.setEnabled(False)

    def layoutWidgets(self) -> None:
        layout_path = QtWidgets.QHBoxLayout()
        layout_path.addWidget(QtWidgets.QLabel("Basename:"))
        layout_path.addWidget(self.lineedit_path)
        layout_path.addWidget(self.button_path)

        layout_isotopes = QtWidgets.QHBoxLayout()
        layout_isotopes.addWidget(self.check_isotopes)
        layout_isotopes.addWidget(self.combo_isotopes)

        layout_preview = QtWidgets.QHBoxLayout()
        layout_preview.addWidget(QtWidgets.QLabel("Preview:"))
        layout_preview.addWidget(self.lineedit_preview)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_path)
        layout_main.addLayout(layout_isotopes)
        layout_main.addWidget(self.check_layers)
        layout_main.addLayout(layout_preview)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

    def accept(self) -> None:
        if not self.isComplete():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Format",
                "Unable to export to this format.\n"
                f"Valid formats: {' '.join(ExportDialog.VALID_FORMATS)}",
            )
            return
        super().accept()

    def buttonPath(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export",
            self.lineedit_path.text(),
            "CSV files(*.csv);;Numpy archives(*.npz);;"
            "Rectilinear VTKs(*.vtr);;All files(*)",
            "All files(*)",
            QtWidgets.QFileDialog.DontConfirmOverwrite,
        )
        if path:
            self.lineedit_path.setText(path)
            self.changed()

    def changed(self) -> None:
        self.contextEnable()
        self.drawPreview()

    def contextEnable(self) -> None:
        ext = os.path.splitext(self.lineedit_path.text())[1].lower()

        if len(self.isotopes) == 1:
            self.check_isotopes.setEnabled(False)
            self.check_isotopes.setChecked(False)
        elif ext in [".npz", ".vtr"]:
            if self.check_isotopes.isEnabled():
                self.check_isotopes.setEnabled(False)
                self.check_isotopes.setChecked(True)
        else:
            self.check_isotopes.setEnabled(True)

        self.combo_isotopes.setEnabled(not self.check_isotopes.isChecked())

        if self.layers == 1:
            self.check_layers.setEnabled(False)
            self.check_layers.setChecked(False)
        elif ext in [".npz", ".vtr"]:
            if self.check_layers.isEnabled():
                self.check_layers.setEnabled(False)
                self.check_layers.setChecked(True)
        else:
            self.check_layers.setEnabled(True)

    def drawPreview(self) -> None:
        path = self.getPath(
            self.names[0] if len(self.names) > 1 else None,
            isotope=self.combo_isotopes.currentText()
            if self.check_isotopes.isEnabled()
            else None,
            layer=1 if self.check_layers.isChecked() else None,
        )

        if not os.path.isdir(path):
            path = os.path.basename(path)
        self.lineedit_preview.setText(path)

    def getPath(self, name: str = None, isotope: str = None, layer: int = None) -> str:
        path, ext = os.path.splitext(self.lineedit_path.text())
        if name is not None:
            path += f"_{name}"
        if isotope is not None:
            path += f"_{isotope}"
        if layer is not None:
            path += f"_{layer}"
        return f"{path}{ext}"

    def isComplete(self) -> bool:
        path, ext = os.path.splitext(self.lineedit_path.text())
        return len(path) > 0 and ext.lower() in ExportDialog.VALID_FORMATS
