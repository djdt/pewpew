import os.path

from PyQt5 import QtWidgets

from pewpew.ui.widgets.overwritefileprompt import OverwriteFilePrompt
from pewpew.lib.laser import Laser

from typing import List, Tuple


class ExportDialog(QtWidgets.QDialog):
    def __init__(
        self,
        path: str,
        name: str,
        nnames: int,
        nlayers: int,
        options: QtWidgets.QWidget,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.path = path
        self.name = name

        self.setWindowTitle("Export")

        self.options = options

        self.check_isotopes = QtWidgets.QCheckBox("Export all isotopes.")
        if nnames < 2:
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
                    name=self.name
                    if self.check_isotopes.isChecked()
                    and self.check_isotopes.isEnabled()
                    else None
                )
            )
        )

    def _get_name(self) -> str:
        return self.name

    def _generate_path(
        self, laser: Laser = None, name: str = None, layer: int = None
    ) -> str:
        path, ext = os.path.splitext(self.path)
        if name is not None:
            path += f"_{name}"
        if layer is not None:
            path += f"_{layer}"
        return path + ext

    def generate_paths(
        self, laser: Laser, prompt: QtWidgets.QMessageBox = None
    ) -> List[Tuple[str, str, int]]:
        paths = []
        if self.check_isotopes.isChecked() and self.check_isotopes.isEnabled():
            if prompt is None:
                prompt = OverwriteFilePrompt(parent=self)
            for name in laser.names():
                path = self._generate_path(laser, name)
                if path == "" or not prompt.promptOverwrite(path):
                    continue
                paths.append((path, name, -1))
        else:
            if prompt is None:
                prompt = OverwriteFilePrompt(show_all_buttons=False, parent=self)
            path = self._generate_path(laser, None)
            if path != "" and prompt.promptOverwrite(path):
                paths.append((path, self._get_name(), -1))
        return paths
