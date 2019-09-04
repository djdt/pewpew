from PySide2 import QtCore, QtWidgets

from pewpew.widgets.dialogs import ApplyDialog


class Tool(ApplyDialog):
    mouseSelectFinished = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout().insertLayout(0, self.layout_top)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)

    # @QtCore.Slot("QWidget*")
    # def mouseSelectFinished(self, widget: QtWidgets.QWidget) -> None:
    #     self.mouseSelectFinished.emit()
    #     if widget is not None and hasattr(widget, "laser"):
    #         self.dock = widget

    #     self.dockarea.mouseSelectFinished.disconnect(self.mouseSelectFinished)
    #     self.activateWindow()
    #     self.setFocus(QtCore.Qt.OtherFocusReason)
    #     self.show()
    #     self.draw()

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() in [
            QtCore.Qt.Key_Escape,
            QtCore.Qt.Key_Enter,
            QtCore.Qt.Key_Return,
        ]:
            return
        if event.key() == QtCore.Qt.Key_F5:
            self.draw()
        super().keyPressEvent(event)
