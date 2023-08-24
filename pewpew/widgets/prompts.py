from pathlib import Path

from PySide6 import QtCore, QtWidgets

from typing import List


class DetailedError(QtWidgets.QMessageBox):
    def __init__(
        self,
        level: QtWidgets.QMessageBox.Icon,
        title: str = "Error",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(level, title, message, QtWidgets.QMessageBox.NoButton, parent)

        self.setDetailedText(detailed_message)
        textedit = self.findChildren(QtWidgets.QTextEdit)
        if textedit[0] is not None:
            textedit[0].setFixedSize(640, 320)

    @staticmethod
    def info(
        title: str = "Info",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QMessageBox.StandardButton:
        return DetailedError(
            QtWidgets.QMessageBox.Information, title, message, detailed_message, parent
        ).exec()

    @staticmethod
    def warning(
        title: str = "Warning",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QMessageBox.StandardButton:
        return DetailedError(
            QtWidgets.QMessageBox.Warning, title, message, detailed_message, parent
        ).exec()

    @staticmethod
    def critical(
        title: str = "Critical",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QMessageBox.StandardButton:
        return DetailedError(
            QtWidgets.QMessageBox.Critical, title, message, detailed_message, parent
        ).exec()


class NonModalMessageBox(QtWidgets.QMessageBox):
    def __init__(
        self,
        level: QtWidgets.QMessageBox.Icon,
        title: str = "Error",
        message: str = "",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(level, title, message, QtWidgets.QMessageBox.NoButton, parent)
        self.setWindowModality(QtCore.Qt.NonModal)

    @staticmethod
    def info(
        title: str = "Info",
        message: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QMessageBox:
        dlg = NonModalMessageBox(
            QtWidgets.QMessageBox.Information, title, message, parent
        ).open()
        return dlg

    @staticmethod
    def warning(
        title: str = "Warning",
        message: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QMessageBox:
        dlg = NonModalMessageBox(
            QtWidgets.QMessageBox.Warning, title, message, parent
        ).open()
        return dlg

    @staticmethod
    def critical(
        title: str = "Critical",
        message: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QMessageBox:
        dlg = NonModalMessageBox(
            QtWidgets.QMessageBox.Critical, title, message, parent
        ).open()
        return dlg


class OverwriteFilePrompt(QtWidgets.QMessageBox):
    def __init__(self, show_all_buttons: bool = True, parent: QtWidgets.QWidget | None = None):
        self.yes_to_all = False
        self.no_to_all = False

        buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        if show_all_buttons:
            buttons |= QtWidgets.QMessageBox.YesToAll | QtWidgets.QMessageBox.NoToAll

        super().__init__(
            QtWidgets.QMessageBox.Warning, "Overwrite File?", "", buttons, parent
        )

    def promptOverwrite(self, path: str | Path) -> bool:
        if isinstance(path, str):  # pragma: no cover
            path = Path(path)

        if self.yes_to_all:
            return True
        if self.no_to_all:
            return False

        if path.exists():
            self.setText(
                f"The file '{path.name}' already exists. Do you wish to overwrite it?"
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

    def promptOverwriteSingleFile(
        self, path: str | Path, parent: QtWidgets.QWidget | None = None
    ) -> bool:
        if isinstance(path, str):  # pragma: no cover
            path = Path(path)

        return OverwriteFilePrompt(
            show_all_buttons=False, parent=parent
        ).promptOverwrite(path)


class OverwriteFilesPrompt(QtWidgets.QMessageBox):
    def __init__(self, paths: List[Path], parent: QtWidgets.QWidget | None = None):
        buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        super().__init__(
            QtWidgets.QMessageBox.Warning,
            "Overwrite multiple files?",
            "\n".join(p.name for p in paths),
            buttons,
            parent,
        )
