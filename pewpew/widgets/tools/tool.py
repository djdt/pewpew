from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.graphics.imageitems import SnapImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.overlayitems import ColorBarOverlay

from pewpew.widgets.views import TabView, TabViewWidget


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
        view: TabView | None = None,
    ):
        super().__init__(editable=False, view=view)
        self._shown = False

        self.graphics = LaserGraphicsView(item.options, parent=self)
        self.colorbar = ColorBarOverlay(
            [], 0, 1, font=item.options.font, color=item.options.font_color
        )
        self.graphics.addOverlayItem(self.colorbar)
        self.graphics.setMouseTracking(True)

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

        layout_graphics = QtWidgets.QVBoxLayout()
        layout_graphics.addWidget(self.graphics, 1)
        self.box_graphics.setLayout(layout_graphics)

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
        if not self.graphics.underMouse():
            return

        action_copy_image = QtGui.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy Scene &Image", self
        )
        action_copy_image.setStatusTip("Copy scene to clipboard.")
        action_copy_image.triggered.connect(self.graphics.copyToClipboard)

        menu = QtWidgets.QMenu(self)
        menu.addAction(action_copy_image)
        menu.popup(event.globalPos())
        event.accept()

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

    def reject(self) -> None:
        self.view.requestClose(self.index)

    def onFirstShow(self) -> None:
        self.graphics.zoomReset()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if not self._shown:
            self.onFirstShow()
            self._shown = True

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)
