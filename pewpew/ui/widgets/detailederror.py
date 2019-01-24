from PyQt5 import QtWidgets


class DetailedError(QtWidgets.QMessageBox):
    def __init__(
        self,
        level: QtWidgets.QMessageBox.Icon,
        title: str = "Error",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget = None,
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
        parent: QtWidgets.QWidget = None,
    ) -> QtWidgets.QMessageBox.StandardButton:
        return DetailedError(
            QtWidgets.QMessageBox.Information, title, message, detailed_message, parent
        ).exec()

    @staticmethod
    def warning(
        title: str = "Warning",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget = None,
    ) -> QtWidgets.QMessageBox.StandardButton:
        return DetailedError(
            QtWidgets.QMessageBox.Warning, title, message, detailed_message, parent
        ).exec()

    @staticmethod
    def critical(
        title: str = "Critical",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget = None,
    ) -> QtWidgets.QMessageBox.StandardButton:
        return DetailedError(
            QtWidgets.QMessageBox.Critical, title, message, detailed_message, parent
        ).exec()
