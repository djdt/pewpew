from PySide2 import QtCore, QtWidgets

from pewpew.widgets.views import _ViewWidget


class ToolWidget(_ViewWidget):
    def __init__(self, widget: _ViewWidget):
        super().__init__(widget.view, editable=False)
        self.widget = widget

        self.label_current = QtWidgets.QLabel()

        self.button_select = QtWidgets.QPushButton("Select &Image")
        self.button_select.pressed.connect(self.startMouseSelect)

        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout_top.addWidget(self.label_current, 0, QtCore.Qt.AlignRight)
        self.layout_top.addWidget(self.button_select, 0, QtCore.Qt.AlignRight)

        self.layout_main = QtWidgets.QVBoxLayout()

        self.layout_buttons = QtWidgets.QHBoxLayout()

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_top)
        layout.addLayout(self.layout_main)
        layout.addLayout(self.layout_buttons)
        self.setLayout(layout)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)

    def startMouseSelect(self) -> None:
        self.viewspace.mouseSelectStart(self)
        self.widget.setActive()

    def endMouseSelect(self) -> None:
        self.viewspace.mouseSelectEnd(self)
        self.widgetChanged()
        self.setActive()

    def widgetChanged(self) -> None:
        pass

    def isComplete(self) -> bool:
        return True

    @QtCore.Slot()
    def completeChanged(self) -> None:
        pass

    def transform(self, **kwargs) -> None:
        if hasattr(self.widget, "transform"):
            self.widget.transform(**kwargs)
        self.widgetChanged()
