import os.path

from PyQt5 import QtWidgets

from pewpew.lib.laser import LaserData

from typing import List, Tuple, Union


class ExportDialog(QtWidgets.QDialog):
    def __init__(
        self,
        path: str,
        isotope: str,
        nisotopes: int,
        nlayers: int,
        options: QtWidgets.QWidget,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.path = path
        self.isotope = isotope

        self.setWindowTitle("Export")

        self.options = options

        self.check_isotopes = QtWidgets.QCheckBox("Export all isotopes.")
        if nisotopes < 2:
            self.check_isotopes.setEnabled(False)
        self.check_isotopes.stateChanged.connect(self._update)
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
        self._update()

    def layoutWidgets(self) -> None:
        layout_preview = QtWidgets.QHBoxLayout()
        layout_preview.addWidget(QtWidgets.QLabel("Preview:"))
        layout_preview.addWidget(self.lineedit_preview)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addWidget(self.options)
        layout_main.addWidget(self.check_isotopes)
        # layout_main.addWidget(self.check_layers)
        layout_main.addLayout(layout_preview)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

    def _update(self) -> None:
        self.lineedit_preview.setText(
            os.path.basename(
                self._generate_path(
                    isotope=self.isotope
                    if self.check_isotopes.isChecked()
                    and self.check_isotopes.isEnabled()
                    else None
                )
            )
        )

    def _generate_path(
        self, laser: LaserData = None, isotope: str = None, layer: int = None
    ) -> str:
        path, ext = os.path.split(self.path)
        if isotope is not None:
            path += f"_{isotope}"
        if layer is not None:
            path += f"_{layer}"
        return path + ext

    def generate_paths(
        self, laser: LaserData, prompt_overwrite: bool = True
    ) -> Tuple[List[Tuple[str, str, int]], bool]:
        paths = []
        isotopes: Union[List[None], List[str]] = [None]
        if self.check_isotopes.isChecked() and self.check_isotopes.isEnabled():
            isotopes = laser.isotopes()
        layers = [None]
        # layers = laser.layers() if self.check_layers.isChecked() else [None]
        for isotope in isotopes:
            for layer in layers:
                path = self._generate_path(laser, isotope, layer)
                if path == "":
                    continue
                if prompt_overwrite and os.path.exists(path):
                    result = QtWidgets.QMessageBox.warning(
                        self,
                        "Overwrite File?",
                        f'The file "{os.path.basename(path)}" '
                        "already exists. Do you wish to overwrite it?",
                        QtWidgets.QMessageBox.Yes
                        | QtWidgets.QMessageBox.YesToAll
                        | QtWidgets.QMessageBox.No,
                    )
                    if result == QtWidgets.QMessageBox.No:
                        continue
                    elif result == QtWidgets.QMessageBox.YesToAll:
                        prompt_overwrite = False
                # Fill in defaults
                if isotope is None:
                    isotope = self.isotope
                if layer is None:
                    layer = 1
                paths.append((path, isotope, layer))
                # self._export(name + ext, isotope, layer, laser)
        return paths, prompt_overwrite
