from PySide6 import QtCore, QtGui, QtWidgets

import logging

from pewpew.actions import qAction, qToolButton

from typing import List, Optional


logger = logging.getLogger(__name__)


class ViewSpace(QtWidgets.QSplitter):
    """Splittable viewspace.

    See Also:
        `:class:pewpew.widgets.views.View`
        `:class:pewpew.widgets.views._ViewWidget`
    """

    activeViewChanged = QtCore.Signal()
    numViewsChanged = QtCore.Signal()
    numTabsChanged = QtCore.Signal()

    def __init__(
        self,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(orientation, parent)
        self.active_view: Optional["View"] = None
        self.views: List[View] = []

        self.action_split_horz = qAction(
            "view-split-left-right",
            "Split &Vertical",
            "Split the viewspace vertically.",
            self.splitActiveHorizontal,
        )
        self.action_split_vert = qAction(
            "view-split-top-bottom",
            "Split &Horizontal",
            "Split the viewspace horizontally.",
            self.splitActiveVertical,
        )
        self.action_close_view = qAction(
            "view-close", "Close View", "Closes the current view.", self.closeActiveView
        )
        # self.action_close_others = QtGui.QAction(
        #     QtGui.QIcon.fromTheme("view-right-close"), "Close Other Views"
        # )
        # self.action_close_view.triggered.connect(self.closeOtherViews)

        self.addWidget(self.createView())

    def activeView(self) -> "View":
        """Return the view last interacted with."""
        if self.active_view is None:
            view = self.views[0]
            self.setActiveView(view)
            assert self.active_view is not None

        return self.active_view

    def activeWidget(self) -> Optional[QtWidgets.QWidget]:
        """Return the tabbed widget last interacted with."""
        view = self.activeView()
        widget = view.activeWidget()
        return widget

    def setActiveView(self, view: "View") -> None:
        """Set the active view."""
        if self.active_view == view:
            return
        self.active_view = view

        self.active_view.viewActivated.emit()
        self.activeViewChanged.emit()
        logger.debug(f"Active view changed: {view}")

    def countViewTabs(self) -> int:
        """Total number of tabbed widgets."""
        widgets = 0
        for view in self.views:
            widgets += view.tabs.count()
        return widgets

    def closeActiveView(self) -> None:
        """Close the active view.."""
        self.closeView(self.activeView())
        self.active_view = None

    # def closeOtherViews(self) -> None:
    #     active_view = self.activeView()
    #     for view in self.views:
    #         if view != active_view:
    #             self.closeView(view)

    def closeView(self, view: "View") -> None:
        """Closes the given `view` and collaspes splitters."""
        if view is None:
            return

        if len(self.views) == 1:
            return

        splitter = view.parent()
        if splitter is None:
            return

        self.views.remove(view)
        self.numViewsChanged.emit()

        view.deleteLater()
        view.setParent(None)

        assert splitter.count() == 1

        if splitter != self:
            parent_splitter = splitter.parent()
            if parent_splitter is not None:
                index = parent_splitter.indexOf(splitter)
                sizes = parent_splitter.sizes()
                parent_splitter.insertWidget(index, splitter.widget(0))
                splitter.deleteLater()
                splitter.setParent(None)
                parent_splitter.setSizes(sizes)
        # Doesn't seem to enter here?
        elif isinstance(splitter.widget(0), QtWidgets.QSplitter):
            child_splitter = splitter.widget(0)
            sizes = child_splitter.sizes()
            splitter.setOrientation(child_splitter.orientation())
            splitter.addWidget(child_splitter.widget(0))
            splitter.addWidget(child_splitter.widget(0))
            child_splitter.deleteLater()
            child_splitter.setParent(None)
            splitter.setSizes(sizes)

    def createView(self) -> "View":
        """Create a new view.

        Does not split! Use splitView.
        """
        view = View(self)
        view.numTabsChanged.connect(self.numTabsChanged)
        self.views.append(view)
        self.numViewsChanged.emit()
        self.setActiveView(view)
        return view

    def splitActiveHorizontal(self) -> None:
        self.splitView(None, QtCore.Qt.Horizontal)

    def splitActiveVertical(self) -> None:
        self.splitView(None, QtCore.Qt.Vertical)

    def splitView(
        self,
        view: "View" = None,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
    ) -> None:
        """Splits the view `view` in two.

        Args:
            view: view to split
            orientation: direction of split
        """
        if view is None:
            view = self.activeView()

        splitter = view.parent()
        index = splitter.indexOf(view)

        if splitter.count() == 1:
            size = (splitter.sizes()[0] - splitter.handleWidth()) / 2.0
            splitter.setOrientation(orientation)
            new_view = self.createView()
            self.addWidget(new_view)
            splitter.insertWidget(index - 1, new_view)
            splitter.setSizes([size, size])
        else:
            new_splitter = QtWidgets.QSplitter(orientation)
            new_splitter.setChildrenCollapsible(False)

            sizes = splitter.sizes()

            new_splitter.addWidget(view)
            new_view = self.createView()
            new_splitter.addWidget(new_view)

            splitter.insertWidget(index, new_splitter)

            splitter.setSizes(sizes)
            new_size = (sum(new_splitter.sizes()) - new_splitter.handleWidth()) / 2.0
            new_splitter.setSizes([new_size, new_size])

    def refresh(self, visible: bool = False) -> None:
        """Resfresh all views"""
        for view in self.views:
            view.refresh(visible)

    # Events
    @QtCore.Slot("QWidget*")
    def mouseSelectStart(self, callback_widget: QtWidgets.QWidget) -> None:
        for view in self.views:
            for widget in view.widgets():
                widget.mouseSelectStart(callback_widget)

    @QtCore.Slot("QWidget*")
    def mouseSelectEnd(self, callback_widget: QtWidgets.QWidget) -> None:
        for view in self.views:
            for widget in view.widgets():
                widget.mouseSelectEnd(callback_widget)


class View(QtWidgets.QWidget):
    """Class to hold tabbed widgets in a ViewSpace.

    See Also:
        `:class:pewpew.widgets.views.ViewSpace`
        `:class:pewpew.widgets.views.ViewTabBar`
        `:class:pewpew.widgets.views._ViewWidget`
    """

    numTabsChanged = QtCore.Signal()
    viewActivated = QtCore.Signal()
    activeWidgetChanged = QtCore.Signal()

    icon_modified = QtGui.QIcon.fromTheme("document-save")

    def __init__(self, viewspace: ViewSpace):
        super().__init__(viewspace)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.viewspace = viewspace

        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.stack = QtWidgets.QStackedWidget()
        self.scroll_area.setWidget(self.stack)

        self.tabs = ViewTabBar(self)
        self.tabs.setDrawBase(False)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        self.tabs.currentChanged.connect(self.activeWidgetChanged)
        self.tabs.tabMoved.connect(self.moveWidget)
        self.tabs.tabCloseRequested.connect(self.requestClose)
        self.tabs.tabTextChanged.connect(self.renameWidget)

        self.stack.installEventFilter(self)
        self.tabs.installEventFilter(self)

        self.titlebar = ViewTitleBar(self.tabs, self)

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.titlebar, 0)
        layout.addWidget(self.scroll_area, 1)
        self.setLayout(layout)

    # Stack
    def activeWidget(self) -> Optional["_ViewWidget"]:
        """The current visable tabbed widget."""
        if self.stack.count() == 0:
            return None
        return self.stack.widget(self.stack.currentIndex())

    def setActiveWidget(self, index: int) -> None:
        if self.stack.currentIndex() == index:
            return
        self.tabs.setCurrentIndex(index)

    def moveWidget(self, ifrom: int, ito: int) -> None:
        """Move tab from index `ifrom` to `ito`."""
        self.stack.insertWidget(ito, self.stack.widget(ifrom))

    def renameWidget(self, index: int, text: str) -> None:
        """Rename a tabbed widget."""
        self.stack.widget(index).rename(text)

    def widgets(self) -> List["_ViewWidget"]:
        """List of all tabbed widgets."""
        return [self.stack.widget(i) for i in range(self.stack.count())]

    # Tabs
    def addTab(self, text: str, widget: "_ViewWidget") -> int:
        """Add a new tabbed widget.

        Args:
            text: tab text
            widget: widget to add

        Returns:
            index of new tab
        """
        index = self.tabs.addTab(text)
        self.stack.insertWidget(index, widget)
        self.setTabModified(index, widget.modified)
        self.numTabsChanged.emit()
        return index

    def insertTab(self, index: int, text: str, widget: "_ViewWidget") -> int:
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

        self.setTabModified(index, widget.modified)
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

    def activate(self) -> None:
        """Set the view as the active view in it's viewspace."""
        self.viewspace.setActiveView(self)

    # Events
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # pragma: no cover
        """Accepts tabbar drags."""
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # pragma: no cover
        """Accepts tabbar drops."""
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            self.tabs.dropEvent(event)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Filter to set as active view on mouse interaction."""
        if obj and event.type() in [
            QtCore.QEvent.MouseButtonPress,
            QtCore.QEvent.Scroll,
        ]:
            self.activate()
        return False


class ViewTabBar(QtWidgets.QTabBar):
    """The tabbar for views.

    Implements closing and drag-drop of tabs."""

    tabTextChanged = QtCore.Signal(int, str)

    def __init__(self, view: View, parent: QtWidgets.QWidget = None):
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

    def actionCloseAll(self) -> None:
        for _ in range(self.count()):
            self.tabCloseRequested.emit(0)

    def actionCloseOthers(self) -> None:
        index = self.action_close_others.data()
        self.moveTab(index, 0)
        for _ in range(1, self.count()):
            self.tabCloseRequested.emit(1)

    def setTabText(self, index: int, text: str) -> None:
        if text != "" and text != self.tabText(index):
            super().setTabText(index, text)
            self.tabTextChanged.emit(index, text)

    def tabRenameDialog(self, index: int) -> QtWidgets.QDialog:
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
        elif ok and isinstance(event.source(), ViewTabBar):
            text = event.source().tabText(src)
            widget = event.source().view.stack.widget(src)

            event.source().view.removeTab(src)

            index = self.view.insertTab(dest, text, widget)
            self.setCurrentIndex(index)
            widget.activate()
            event.acceptProposedAction()
        else:
            event.rejectProposedAction()


class ViewTitleBar(QtWidgets.QWidget):
    """Titlebar with buttons for view."""

    def __init__(self, tabs: ViewTabBar, view: View):
        super().__init__(view)
        self.view = view

        self.split_button = qToolButton("view-split-left-right")
        self.split_button.addAction(self.view.viewspace.action_split_horz)
        self.split_button.addAction(self.view.viewspace.action_split_vert)
        self.split_button.addAction(self.view.viewspace.action_close_view)
        # self.split_button.addAction(self.view.viewspace.action_close_others)

        self.split_button.installEventFilter(self.view)

        # Layout the widgets
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs, 1)
        layout.addWidget(self.split_button)
        self.setLayout(layout)


class _ViewWidget(QtWidgets.QWidget):
    """Base class for widgets intending to be used in a view.

    See Also:
        `:class:pewpew.widgets.views.View`
        `:class:pewpew.widgets.views.ViewSpace`
    """

    refreshed = QtCore.Signal()

    def __init__(self, view: View = None, editable: bool = True):
        super().__init__(view)

        self.view = view

        self.editable = editable
        self._modified = False

    @property
    def viewspace(self) -> Optional[ViewSpace]:
        return self.view.viewspace if self.view is not None else None

    @property
    def index(self) -> int:
        return self.view.stack.indexOf(self)

    @property
    def name(self) -> str:
        return self.view.tabs.tabText(self.index)

    @property
    def modified(self) -> bool:
        return self._modified

    @modified.setter
    def modified(self, modified: bool) -> None:
        self._modified = modified
        self.view.setTabModified(self.index, modified)

    def refresh(self) -> None:  # pragma: no cover
        self.refreshed.emit()

    def rename(self, text: str) -> None:  # pragma: no cover
        pass

    def requestClose(self) -> bool:
        return True

    def activate(self) -> None:
        self.view.activate()
        self.view.setActiveWidget(self.index)

    @QtCore.Slot("QWidget*")
    def mouseSelectStart(self, callback_widget: QtWidgets.QWidget) -> None:
        pass

    @QtCore.Slot("QWidget*")
    def mouseSelectEnd(self, callback_widget: QtWidgets.QWidget) -> None:
        pass

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj and event.type() in [
            QtCore.QEvent.MouseButtonPress,
            QtCore.QEvent.Scroll,
        ]:
            self.view.activate()
            self.activate()
        return False
