from PyQt5 import QtCore, QtWidgets


# TODO We should redo this with the layout built in
class ApplyDialog(QtWidgets.QDialog):

    applyPressed = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.layout_form = QtWidgets.QFormLayout()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply,
            self,
        )
        self.button_box.clicked.connect(self.buttonClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_form)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def buttonClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.Apply:
            if self.complete():
                self.apply()
                self.applyPressed.emit(self)
            else:
                self.error()
        elif sb == QtWidgets.QDialogButtonBox.Ok:
            if self.complete():
                self.accept()
            else:
                self.error()
        else:
            self.reject()

    def apply(self) -> None:
        pass

    def complete(self) -> bool:
        return True

    def error(self) -> None:
        pass
