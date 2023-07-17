import logging
from typing import List

from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.actions import qAction

logger = logging.getLogger(__name__)


class TabView(QtWidgets.QWidget):
    """Class to hold tabbed widgets, with tabs at the top.

    See Also:
        `:class:pewpew.widgets.views.ViewSpace`
        `:class:pewpew.widgets.views.ViewTabBar`
        `:class:pewpew.widgets.views.TabViewWidget`
    """

    numTabsChanged = QtCore.Signal()
    viewActivated = QtCore.Signal()
    activeWidgetChanged = QtCore.Signal()

    icon_modified = QtGui.QIcon.fromTheme("document-save")

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.stack = QtWidgets.QStackedWidget()
        self.scroll_area.setWidget(self.stack)

        self.tabs = TabViewBar(self)
        self.tabs.setDrawBase(False)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        self.tabs.currentChanged.connect(self.activeWidgetChanged)
        self.tabs.tabMoved.connect(self.moveWidget)
        self.tabs.tabCloseRequested.connect(self.requestClose)
        self.tabs.tabTextChanged.connect(self.renameWidget)

        self.stack.installEventFilter(self)
        self.tabs.installEventFilter(self)

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs, 0)
        layout.addWidget(self.scroll_area, 1)
        self.setLayout(layout)

    # Stack
    def activeWidget(self) -> ["TabViewWidget"]:
        """The current visable tabbed widget."""
        if self.stack.count() == 0:
            return None
        return self.stack.widget(self.stack.currentIndex())

    def setActiveWidget(self, widget: "TabViewWidget") -> None:
        index = self.stack.indexOf(widget)
        if self.stack.currentIndex() != index:
            self.tabs.setCurrentIndex(index)

    def moveWidget(self, ifrom: int, ito: int) -> None:
        """Move tab from index `ifrom` to `ito`."""
        self.stack.insertWidget(ito, self.stack.widget(ifrom))

    def renameWidget(self, index: int, text: str) -> None:
        """Rename a tabbed widget."""
        self.stack.widget(index).rename(text)

    def widgets(self) -> List["TabViewWidget"]:
        """List of all tabbed widgets."""
        return [self.stack.widget(i) for i in range(self.stack.count())]

    # Tabs
    def addTab(self, text: str, widget: "TabViewWidget") -> int:
        """Add a new tabbed widget.

        Args:
            text: tab text
            widget: widget to add

        Returns:
            index of new tab
        """
        return self.insertTab(self.tabs.count(), text, widget)

    def insertTab(self, index: int, text: str, widget: "TabViewWidget") -> int:
        """Add a new tabbed widget at index.

        Args:
            index: add at
            text: tab text
            widget: widget to add

        Returns:
            index of new tab
        """
        index = self.tabs.insertTab(index, text)
        self.stack.insertWidget(index, widget)
        widget.view = self

        self.setTabModified(index, widget.isWindowModified())
        self.numTabsChanged.emit()
        return index

    def removeTab(self, index: int) -> None:
        """Remove tab and widget at index."""
        self.tabs.removeTab(index)
        self.stack.removeWidget(self.stack.widget(index))
        self.numTabsChanged.emit()

    def setTabModified(self, index: int, modified: bool = True) -> None:
        """Shows a modified icon on tab at `index`."""
        icon = self.icon_modified if modified else QtGui.QIcon()
        self.tabs.setTabIcon(index, icon)

    def refresh(self, visible: bool = False) -> None:
        """Resfresh all or `visible` widgets."""
        if visible:
            widget = self.activeWidget()
            if widget is not None:
                widget.refresh()
        else:
            for widget in self.widgets():
                widget.refresh()

    def requestClose(self, index: int) -> None:
        """Try to close the widget at `index`.

        Only closes if widget allows it."""
        if self.stack.widget(index).requestClose():
            self.removeTab(index)

    # Events
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # pragma: no cover
        """Accepts tabbar drags."""
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # pragma: no cover
        """Accepts tabbar drops."""
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            self.tabs.dropEvent(event)


class TabViewBar(QtWidgets.QTabBar):
    """The tabbar for views.

    Implements closing and drag-drop of tabs."""

    tabTextChanged = QtCore.Signal(int, str)

    def __init__(self, view: TabView, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.view = view
        self.drag_start_pos = QtCore.QPoint(0, 0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.setElideMode(QtCore.Qt.ElideLeft)
        self.setUsesScrollButtons(True)
        self.setExpanding(False)
        self.setTabsClosable(True)

        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        self.action_close_all = qAction(
            "view-close",
            "Close All Tabs",
            "Closes all tabs open in this view.",
            self.actionCloseAll,
        )
        self.action_close_others = qAction(
            "view-right-close",
            "Close Other Tabs",
            "Close all tabs but this one.",
            self.actionCloseOthers,
        )
        self.action_open = qAction(
            "document-open", "&Open", "Open new document(s).", self.actionOpen
        )

    def actionCloseAll(self) -> None:
        for _ in range(self.count()):
            self.tabCloseRequested.emit(0)

    def actionCloseOthers(self) -> None:
        index = self.action_close_others.data()
        self.moveTab(index, 0)
        for _ in range(1, self.count()):
            self.tabCloseRequested.emit(1)

    def actionOpen(self) -> None:
        raise NotImplementedError

    def setTabText(self, index: int, text: str) -> None:
        if text != "" and text != self.tabText(index):
            super().setTabText(index, text)
            self.tabTextChanged.emit(index, text)

    def tabRenameDialog(self, index: int) -> QtWidgets.QDialog | None:
        if index == -1 or not self.view.stack.widget(index).editable:
            return
        dlg = QtWidgets.QInputDialog(self)
        dlg.setWindowTitle("Rename")
        dlg.setLabelText("Name:")
        dlg.setTextValue(self.tabText(index))
        dlg.setInputMode(QtWidgets.QInputDialog.TextInput)
        dlg.textValueSelected.connect(lambda s: self.setTabText(index, s))
        dlg.open()
        return dlg

    # def minimumSizeHint(self) -> int:
    #     return 50, 50

    def minimumTabSizeHint(self, index: int) -> QtCore.QSize:
        size = super().minimumTabSizeHint(index)
        return QtCore.QSize(160, size.height())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.drag_start_pos = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # pragma: no cover
        if (
            not event.buttons() & QtCore.Qt.LeftButton
            or (event.position() - self.drag_start_pos).manhattanLength()
            < QtWidgets.QApplication.startDragDistance()
        ):
            return super().mouseMoveEvent(event)
        index = self.tabAt(event.position())
        if index == -1:
            return super().mouseMoveEvent(event)

        rect = self.tabRect(index)
        pixmap = QtGui.QPixmap(rect.size())
        self.render(pixmap, QtCore.QPoint(), QtGui.QRegion(rect))

        mime_data = QtCore.QMimeData()
        mime_data.setData("application/x-pew2tabbar", QtCore.QByteArray().number(index))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mime_data)
        drag.setPixmap(pixmap)
        drag.setDragCursor(
            QtGui.QCursor(QtCore.Qt.DragMoveCursor).pixmap(), QtCore.Qt.MoveAction
        )
        drag.exec_(QtCore.Qt.MoveAction)

    def mouseDoubleClickEvent(self, event: QtGui.QContextMenuEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            index = self.tabAt(event.position())
            self.tabRenameDialog(index)
        else:
            super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        event.accept()
        index = self.tabAt(event.position())
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_close_all)
        if index != -1:
            self.action_close_others.setData(index)
            menu.addAction(self.action_close_others)
        menu.popup(event.globalPos())

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # pragma: no cover
        if event.mimeData().hasFormat("application/x-pew2tabbar"):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # pragma: no cover
        dest = self.tabAt(event.position())
        src, ok = event.mimeData().data("application/x-pew2tabbar").toInt()
        if ok and event.source() == self:
            self.moveTab(src, dest)
            event.acceptProposedAction()
        # elif ok and isinstance(event.source(), TabViewBar):
        #     text = event.source().tabText(src)
        #     widget = event.source().view.stack.widget(src)

        #     event.source().view.removeTab(src)

        #     index = self.view.insertTab(dest, text, widget)
        #     self.setCurrentIndex(index)
        #     widget.activate()
        #     event.acceptProposedAction()
        else:
            event.rejectProposedAction()


class TabViewWidget(QtWidgets.QWidget):
    """Base class for widgets intending to be used in a TabView.

    See Also:
        `:class:pewpew.widgets.views.View`
        `:class:pewpew.widgets.views.ViewSpace`
    """

    refreshed = QtCore.Signal()

    def __init__(self, view: [TabView], editable: bool = True):
        super().__init__(view)

        self.view = view
        self.editable = editable

    @property
    def index(self) -> int:
        return self.view.stack.indexOf(self)

    @property
    def name(self) -> str:
        return self.view.tabs.tabText(self.index)

    def setWindowModified(self, modified: bool) -> None:
        super().setWindowModified(modified)
        self.view.setTabModified(self.index, modified)

    def refresh(self) -> None:  # pragma: no cover
        self.refreshed.emit()

    def rename(self, text: str) -> None:
        raise NotImplementedError

    def requestClose(self) -> bool:
        return True

    def activate(self) -> None:
        self.view.setActiveWidget(self)
