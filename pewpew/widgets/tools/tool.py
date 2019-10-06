from PySide2 import QtCore, QtWidgets

from pewpew.widgets.views import _ViewWidget


class ToolWidget(_ViewWidget):
    mouseSelectStarted = QtCore.Signal("QWidget*")
    mouseSelectEnded = QtCore.Signal("QWidget*")

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        widget = QtWidgets.QWidget()

        self.button_select = QtWidgets.QPushButton("Select &Image")
        self.button_select.pressed.connect(self.startMouseSelect)

        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.button_box.clicked.connect(self.buttonClicked)

        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout_top.addWidget(self.button_select, 0, QtCore.Qt.AlignRight)

        self.layout_main = QtWidgets.QVBoxLayout()

        self.layout_buttons = QtWidgets.QHBoxLayout()
        self.layout_buttons.addWidget(self.button_box)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_top)
        layout.addLayout(self.layout_main)
        layout.addLayout(self.layout_buttons)
        self.setLayout(layout)

        widget.setLayout(layout)
        self.setWidget(widget)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)

    def startMouseSelect(self) -> None:
        self.hide()
        self.mouseSelectStarted.emit(self)

    def endMouseSelect(self) -> None:
        self.mouseSelectEnded.emit(self)
        self.activateWindow()
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self.show()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        raise NotImplementedError

    def refresh(self) -> None:
        pass

    # def keyPressEvent(self, event: QtCore.QEvent) -> None:
    #     if event.key() in [
    #         QtCore.Qt.Key_Escape,
    #         QtCore.Qt.Key_Enter,
    #         QtCore.Qt.Key_Return,
    #     ]:
    #         return
    #     if event.key() == QtCore.Qt.Key_F5:
    #         self.refresh()
    #     super().keyPressEvent(event)

    def buttonClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.Save:
            self.apply()
            self.applyPressed.emit(self)
        elif sb == QtWidgets.QDialogButtonBox.SaveAll:
            self.apply()
            self.applyPressed.emit(self)
            self.accept()
        else:
            self.reject()

    def apply(self) -> None:
        pass

    def isComplete(self) -> bool:
        return True

    @QtCore.Slot()
    def completeChanged(self) -> None:
        enabled = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(enabled)
        self.button_box.button(QtWidgets.QDialogButtonBox.Apply).setEnabled(enabled)


class ToolWidget(QtWidgets.QScrollArea):
    mouseSelectStarted = QtCore.Signal("QWidget*")
    mouseSelectEnded = QtCore.Signal("QWidget*")

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )

        self.button_select = QtWidgets.QPushButton("Select &Image")
        self.button_select.pressed.connect(self.startMouseSelect)

        widget = QtWidgets.QWidget()

        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout_top.addWidget(self.button_select, 0, QtCore.Qt.AlignRight)

        widget.setLayout(self.layout_top)
        self.setWidget(widget)
        # self.layout().insertLayout(0, self.layout_top)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)

    def startMouseSelect(self) -> None:
        self.hide()
        self.mouseSelectStarted.emit(self)

    def endMouseSelect(self) -> None:
        self.mouseSelectEnded.emit(self)
        self.activateWindow()
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self.show()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        raise NotImplementedError

    def refresh(self) -> None:
        pass

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() in [
            QtCore.Qt.Key_Escape,
            QtCore.Qt.Key_Enter,
            QtCore.Qt.Key_Return,
        ]:
            return
        if event.key() == QtCore.Qt.Key_F5:
            self.refresh()
        super().keyPressEvent(event)


class Tool(ApplyDialog):
    mouseSelectStarted = QtCore.Signal("QWidget*")
    mouseSelectEnded = QtCore.Signal("QWidget*")

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )

        self.button_select = QtWidgets.QPushButton("Select &Image")
        self.button_select.pressed.connect(self.startMouseSelect)

        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout_top.addWidget(self.button_select, 0, QtCore.Qt.AlignRight)
        self.layout().insertLayout(0, self.layout_top)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)

    def startMouseSelect(self) -> None:
        self.hide()
        self.mouseSelectStarted.emit(self)

    def endMouseSelect(self) -> None:
        self.mouseSelectEnded.emit(self)
        self.activateWindow()
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self.show()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        raise NotImplementedError

    def refresh(self) -> None:
        pass

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() in [
            QtCore.Qt.Key_Escape,
            QtCore.Qt.Key_Enter,
            QtCore.Qt.Key_Return,
        ]:
            return
        if event.key() == QtCore.Qt.Key_F5:
            self.refresh()
        super().keyPressEvent(event)
