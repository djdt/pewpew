from PyQt5 import QtCore, QtWidgets


class ApplyDialog(QtWidgets.QDialog):

    applyPressed = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply,
            self,
        )
        self.button_box.clicked.connect(self.buttonClicked)

    def buttonClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.Apply:
            self.apply()
            self.applyPressed.emit(self)
        elif sb == QtWidgets.QDialogButtonBox.Ok:
            self.accept()
        else:
            self.reject()

    def apply(self) -> None:
        pass
