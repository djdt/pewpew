from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.widgets.views import TabView, TabViewWidget
from pewpew.graphics.imageitems import SnapImageItem

from typing import Optional


class ToolWidget(TabViewWidget):
    """Base widget for pewpew tools.

    Provides a layout with two group boxes, oriented as per `orientation`.
    One contains controls for the tool, the other a graphical preview.

    Args:
        control_label: text over controls
        graphics_label: text over graphics
        orientation: layout of controls / graphics
        apply_all: add an Apply All button

    Parameters:
        widget: widget being modified
        box_controls: GroupBox for controls
        box_graphics: GroupBox for graphics
    """
    applyPressed = QtCore.Signal()
    applyAllPressed = QtCore.Signal()

    itemModified = QtCore.Signal(SnapImageItem)

    def __init__(
        self,
        item: SnapImageItem,
        control_label: str = "Controls",
        graphics_label: str = "",
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
        apply_all: bool = False,
        view: Optional[TabView] = None,
    ):
        super().__init__(editable=False, view=view)
        
        self.graphics = OverlayGraphicsView()
        self.item = item

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

    # def contextMenuEvent(self, event: QtCore.QEvent) -> None:
    #     event.accept()
    #     action_copy_image = QtWidgets.QAction(
    #         QtGui.QIcon.fromTheme("insert-image"), "Copy To Clipboard", self
    #     )
    #     action_copy_image.setStatusTip("Copy the graphics view to the clipboard.")
    #     action_copy_image.triggered.connect(self.graphics.copyToClipboard)

    #     menu = QtWidgets.QMenu(self)
    #     menu.addAction(action_copy_image)
    #     menu.popup(event.globalPos())

    def accept(self) -> None:  # pragma: no cover
        self.view.requestClose(self.index)

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

    # def onFirstShow(self) -> None:
    #     if self.graphics is None:
    #         return

    #     rect = self.widget.graphics.mapToScene(
    #         self.widget.graphics.viewport().rect()
    #     ).boundingRect()
    #     rect = rect.intersected(self.widget.graphics.sceneRect())
    #     self.graphics.fitInView(rect, QtCore.Qt.KeepAspectRatio)
    #     self.graphics.updateForeground()
    #     self.graphics.invalidateScene()

    def reject(self) -> None:
        self.view.requestClose(self.index)

    # def restoreWidget(self) -> None:
    #     """Remove the tool and replace with it's `widget`."""
    #     self.view.removeTab(self.index)
    #     self.view.insertTab(self.index, self.widget.laserName(), self.widget)
    #     self.widget.modified = self.modified
    #     self.widget.activate()

    # def requestClose(self) -> bool:
    #     self.reject()
    #     return False

    # def showEvent(self, event: QtGui.QShowEvent) -> None:
    #     super().showEvent(event)
    #     if not self._shown:
    #         self.onFirstShow()
    #         self._shown

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)

    # def transform(self, **kwargs) -> None:
    #     if hasattr(self.widget, "transform"):
    #         self.widget.transform(**kwargs)
    #     self.refresh()
