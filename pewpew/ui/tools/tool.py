from PyQt5 import QtCore, QtWidgets


class Tool(QtWidgets.QDialog):

    applyPressed = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout_left = QtWidgets.QVBoxLayout()
        self.layout_right = QtWidgets.QVBoxLayout()
        self.layout_bottom = QtWidgets.QHBoxLayout()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply,
            self,
        )
        self.button_box.clicked.connect(self.buttonClicked)
        self.layout_bottom.addWidget(self.button_box)

        layout_center = QtWidgets.QHBoxLayout()
        layout_center.addLayout(self.layout_left)
        layout_center.addLayout(self.layout_right)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_top)
        layout.addLayout(layout_center)
        layout.addLayout(self.layout_bottom)
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
