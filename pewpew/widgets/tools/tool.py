from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.widgets.views import _ViewWidget


class ToolWidget(_ViewWidget):
    applyPressed = QtCore.Signal()
    applyAllPressed = QtCore.Signal()

    def __init__(
        self,
        widget: _ViewWidget,
        control_label: str = "Controls",
        graphics_label: str = "",
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
        apply_all: bool = False,
    ):
        super().__init__(widget.view, editable=False)
        self.widget = widget
        self.modified = widget.modified
        self._shown = False

        self.graphics: QtWidgets.QGraphicsView = None

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

        self.box_controls = QtWidgets.QGroupBox(control_label)
        self.box_graphics = QtWidgets.QGroupBox(graphics_label)

        self.splitter = QtWidgets.QSplitter(orientation)
        if orientation == QtCore.Qt.Horizontal:
            self.splitter.addWidget(self.box_controls)
            self.splitter.setStretchFactor(0, 0)
            self.splitter.addWidget(self.box_graphics)
            self.splitter.setStretchFactor(1, 1)
        else:
            self.splitter.addWidget(self.box_graphics)
            self.splitter.setStretchFactor(0, 1)
            self.splitter.addWidget(self.box_controls)
            self.splitter.setStretchFactor(1, 0)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.splitter)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        action_copy_image = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy To Clipboard", self
        )
        action_copy_image.setStatusTip("Copy the graphics view to the clipboard.")
        action_copy_image.triggered.connect(self.graphics.copyToClipboard)

        menu = QtWidgets.QMenu(self)
        menu.addAction(action_copy_image)
        menu.popup(event.globalPos())

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

    def onFirstShow(self) -> None:
        if self.graphics is None:
            return

        rect = self.widget.graphics.mapToScene(
            self.widget.graphics.viewport().rect()
        ).boundingRect()
        rect = rect.intersected(self.widget.graphics.sceneRect())
        self.graphics.fitInView(rect, QtCore.Qt.KeepAspectRatio)
        self.graphics.updateForeground()
        self.graphics.invalidateScene()

    def reject(self) -> None:
        self.restoreWidget()

    def restoreWidget(self) -> None:
        self.view.insertTab(self.index, self.widget.laser.name, self.widget)
        self.view.removeTab(self.index)
        self.widget.modified = self.modified
        self.widget.setActive()

    def requestClose(self) -> bool:
        self.reject()
        return False

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if not self._shown:
            self.onFirstShow()
            self._shown

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)

    def transform(self, **kwargs) -> None:
        if hasattr(self.widget, "transform"):
            self.widget.transform(**kwargs)
        self.refresh()
