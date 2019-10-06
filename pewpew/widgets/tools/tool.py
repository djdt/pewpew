from PySide2 import QtCore, QtWidgets

from pewpew.widgets.dialogs import ApplyDialog


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
