from PySide2 import QtCore, QtWidgets

from pewpew.widgets.views import _ViewWidget


class ToolWidget(_ViewWidget):
    def __init__(self, widget: _ViewWidget):
        super().__init__(widget.view, editable=False)
        self.widget = widget

        self.layout_main = QtWidgets.QVBoxLayout()

        self.layout_buttons = QtWidgets.QHBoxLayout()

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_main)
        layout.addLayout(self.layout_buttons)
        self.setLayout(layout)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)

    def isComplete(self) -> bool:
        return True

    @QtCore.Slot()
    def completeChanged(self) -> None:
        pass

    def transform(self, **kwargs) -> None:
        if hasattr(self.widget, "transform"):
            self.widget.transform(**kwargs)
        self.widgetChanged()
