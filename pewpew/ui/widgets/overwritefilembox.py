import os.path

from PyQt5 import QtWidgets


class OverwriteFileMessageBox(QtWidgets.QMessageBox):
    def __init__(self, button_all: bool = True, parent: QtWidgets.QWidget = None):
        self.yes_to_all = False
        self.no_to_all = False

        buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        if button_all:
            buttons |= QtWidgets.QMessageBox.YesToAll | QtWidgets.QMessageBox.NoToAll

        super().__init__(
            QtWidgets.QMessageBox.Warning, "Overwrite File?", "", buttons, parent
        )

    def promptOverwrite(self, path: str) -> bool:
        if self.yes_to_all:
            return True
        if self.no_to_all:
            return False

        if os.path.exists(path):
            self.setText(
                f'The file "{os.path.basename(path)}" '
                "already exists. Do you wish to overwrite it?"
            )
            result = self.exec()
            if result == QtWidgets.QMessageBox.YesToAll:
                self.yes_to_all = True
                return True
            elif result == QtWidgets.QMessageBox.NoToAll:
                self.no_to_all = True
                return False
            elif result == QtWidgets.QMessageBox.Yes:
                return True
            else:
                return False
        return True

    def promptOverwriteSingleFile(path: str, parent: QtWidgets.QWidget = None) -> bool:
        return OverwriteFileMessageBox(button_all=False, parent=parent).promptOverwrite(
            path
        )
