from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.widgets.views import _ViewWidget


class ToolWidget(_ViewWidget):
    applyPressed = QtCore.Signal()
    applyAllPressed = QtCore.Signal()

    def __init__(self, widget: _ViewWidget, apply_all: bool = False):
        super().__init__(widget.view, editable=False)
        self.widget = widget

        self.layout_main = QtWidgets.QVBoxLayout()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply,
            self,
        )
        self.button_box.clicked.connect(self.buttonClicked)

        self.button_apply_all = None
        if apply_all:
            self.button_apply_all = self.button_box.addButton(
                "Apply All", QtWidgets.QDialogButtonBox.ApplyRole
            )
            self.button_apply_all.setIcon(QtGui.QIcon.fromTheme("dialog-ok-apply"))
            self.button_apply_all.clicked.connect(self.applyAll)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_main)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def accept(self) -> None:  # pragma: no cover
        self.restoreWidget()

    def apply(self) -> None:  # pragma: no cover
        pass

    def applyAll(self) -> None:  # pragma: no cover
        pass

    def buttonClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.Apply:
            self.apply()
            self.applyPressed.emit()
        elif sb == QtWidgets.QDialogButtonBox.Ok:
            self.apply()
            self.applyPressed.emit()
            self.accept()
        elif sb == QtWidgets.QDialogButtonBox.Cancel:
            self.reject()

    @QtCore.Slot()
    def completeChanged(self) -> None:
        enabled = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(enabled)
        self.button_box.button(QtWidgets.QDialogButtonBox.Apply).setEnabled(enabled)
        if self.button_apply_all is not None:
            self.button_apply_all.setEnabled(enabled)

    def isComplete(self) -> bool:  # pragma: no cover
        return True

    def reject(self) -> None:
        self.restoreWidget()

    def restoreWidget(self) -> None:
        self.view.insertTab(self.index, self.widget.laser.name, self.widget)
        self.view.removeTab(self.index)
        self.widget.setActive()

    def requestClose(self) -> bool:
        self.reject()
        return False

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)

    def transform(self, **kwargs) -> None:
        if hasattr(self.widget, "transform"):
            self.widget.transform(**kwargs)
        self.refresh()
