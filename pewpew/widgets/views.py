from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.actions import qToolButton

from typing import List


class ViewSpace(QtWidgets.QSplitter):
    numViewsChanged = QtCore.Signal()
    numTabsChanged = QtCore.Signal()

    def __init__(
        self,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(orientation, parent)
        self.active_view: "View" = None
        self.views: List[View] = []

        self.action_split_horz = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-split-left-right"), "Split &Vertical"
        )
        self.action_split_horz.triggered.connect(self.splitActiveHorizontal)
        self.action_split_vert = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-split-top-bottom"), "Split &Horizontal"
        )
        self.action_split_vert.triggered.connect(self.splitActiveVertical)
        self.action_close_view = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-close"), "Close View"
        )
        self.action_close_view.triggered.connect(self.closeActiveView)
        # self.action_close_others = QtWidgets.QAction(
        #     QtGui.QIcon.fromTheme("view-right-close"), "Close Other Views"
        # )
        # self.action_close_view.triggered.connect(self.closeOtherViews)

        self.addWidget(self.createView())

    def activeView(self) -> "View":
        if self.active_view is None:
            view = self.views[0]
            self.active_view = view

        return self.active_view

    def activeWidget(self) -> QtWidgets.QWidget:
        view = self.activeView()
        widget = view.activeWidget()
        return widget

    def setActiveView(self, view: "View") -> None:
        if self.active_view == view:
            return
        if self.active_view is not None:
            self.active_view.setActive(False)
        # if view is not None:
        #     view.setActive(True)
        self.active_view = view

    def countViewTabs(self) -> int:
        widgets = 0
        for view in self.views:
            widgets += view.tabs.count()
        return widgets

    def closeActiveView(self) -> None:
        self.closeView(self.activeView())
        self.active_view = None

    # def closeOtherViews(self) -> None:
    #     active_view = self.activeView()
    #     for view in self.views:
    #         if view != active_view:
    #             self.closeView(view)

    def closeView(self, view: "View") -> None:
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
        view = View(self)
        view.numTabsChanged.connect(self.numTabsChanged)
        self.views.append(view)
        self.numViewsChanged.emit()
        self.setActiveView(view)
        return view

    def splitActiveHorizontal(self) -> None:
        self.splitView()

    def splitActiveVertical(self) -> None:
        self.splitView(None, QtCore.Qt.Vertical)

    def splitView(
        self,
        view: "View" = None,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
    ) -> None:
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
    numTabsChanged = QtCore.Signal()

    icon_modified = QtGui.QIcon.fromTheme("document-save")

    def __init__(self, viewspace: ViewSpace):
        super().__init__(viewspace)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.active = False
        self.viewspace = viewspace

        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.stack = QtWidgets.QStackedWidget()
        self.scroll_area.setWidget(self.stack)

        self.tabs = ViewTabBar(self)
        self.tabs.setDrawBase(False)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        self.tabs.tabMoved.connect(self.moveWidget)
        self.tabs.tabCloseRequested.connect(self.removeTab)
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
    def activeWidget(self) -> "_ViewWidget":
        if self.stack.count() == 0:
            return None
        return self.stack.widget(self.stack.currentIndex())

    def moveWidget(self, ifrom: int, ito: int) -> None:
        self.stack.insertWidget(ito, self.stack.widget(ifrom))

    def renameWidget(self, index: int, text: str) -> None:
        self.stack.widget(index).rename(text)

    def widgets(self) -> List["_ViewWidget"]:
        return [self.stack.widget(i) for i in range(self.stack.count())]

    # Tabs
    def addTab(self, text: str, widget: "_ViewWidget") -> int:
        index = self.tabs.addTab(text)
        self.stack.insertWidget(index, widget)
        self.numTabsChanged.emit()
        return index

    def insertTab(self, index: int, text: str, widget: "_ViewWidget") -> int:
        index = self.tabs.insertTab(index, text)
        self.stack.insertWidget(index, widget)
        self.numTabsChanged.emit()
        return index

    def removeTab(self, index: int) -> None:
        self.tabs.removeTab(index)
        self.stack.removeWidget(self.stack.widget(index))
        self.numTabsChanged.emit()

    def setTabModified(self, index: int, modified: bool = True) -> None:
        icon = self.icon_modified if modified else QtGui.QIcon()
        self.tabs.setTabIcon(index, icon)

    def refresh(self, visible: bool = False) -> None:
        if visible:
            widget = self.activeWidget()
            if widget is not None:
                widget.refresh()
        else:
            for widget in self.widgets():
                widget.refresh()

    def setActive(self, active: bool) -> None:
        if active:
            self.viewspace.setActiveView(self)
        self.active = active

    # Events
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # pragma: no cover
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # pragma: no cover
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            self.tabs.dropEvent(event)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj and event.type() == QtCore.QEvent.MouseButtonPress:
            self.setActive(True)
        return False


class ViewTabBar(QtWidgets.QTabBar):
    tabTextChanged = QtCore.Signal(int, str)

    def __init__(self, view: View, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.view = view
        self.drag_start_pos = QtCore.QPoint(0, 0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.setElideMode(QtCore.Qt.ElideRight)
        self.setExpanding(False)
        self.setTabsClosable(True)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        self.tabBarDoubleClicked.connect(self.tabRenameDialog)

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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # pragma: no cover
        if (
            not event.buttons() & QtCore.Qt.LeftButton
            or (event.pos() - self.drag_start_pos).manhattanLength()
            < QtWidgets.QApplication.startDragDistance()
        ):
            return super().mouseMoveEvent(event)
        index = self.tabAt(event.pos())
        if index == -1:
            return super().mouseMoveEvent(event)

        rect = self.tabRect(index)
        pixmap = QtGui.QPixmap(rect.size())
        self.render(pixmap, QtCore.QPoint(), QtGui.QRegion(rect))

        mime_data = QtCore.QMimeData()
        mime_data.setData(
            "application/x-pewpewtabbar", QtCore.QByteArray().number(index)
        )

        drag = QtGui.QDrag(self)
        drag.setMimeData(mime_data)
        drag.setPixmap(pixmap)
        drag.setDragCursor(
            QtGui.QCursor(QtCore.Qt.DragMoveCursor).pixmap(), QtCore.Qt.MoveAction
        )
        drag.exec_(QtCore.Qt.MoveAction)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # pragma: no cover
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # pragma: no cover
        dest = self.tabAt(event.pos())
        src, ok = event.mimeData().data("application/x-pewpewtabbar").toInt()
        if ok and event.source() == self:
            self.moveTab(src, dest)
        elif ok and isinstance(event.source(), ViewTabBar):
            text = event.source().tabText(src)
            widget = event.source().view.stack.widget(src)

            event.source().view.removeTab(src)

            index = self.view.insertTab(dest, text, widget)
            self.setCurrentIndex(index)
        else:
            return

        event.acceptProposedAction()


class ViewTitleBar(QtWidgets.QWidget):
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
    def __init__(self, view: View, editable: bool = True):
        super().__init__(view)
        self.view = view
        self.viewspace = view.viewspace

        self.editable = editable

    @property
    def index(self) -> int:
        return self.view.stack.indexOf(self)

    @property
    def name(self) -> str:
        return self.view.tabs.tabText(self.index)

    def refresh(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def rename(self, text: str) -> None:  # pragma: no cover
        pass

    def setActive(self) -> None:
        self.view.tabs.setCurrentIndex(self.index)

    def setModified(self, modified: bool) -> None:
        self.view.setTabModified(self.index, modified)

    @QtCore.Slot("QWidget*")
    def mouseSelectStart(self, callback_widget: QtWidgets.QWidget) -> None:
        pass

    @QtCore.Slot("QWidget*")
    def mouseSelectEnd(self, callback_widget: QtWidgets.QWidget) -> None:
        pass

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj and event.type() == QtCore.QEvent.MouseButtonPress:  # pragma: no cover
            self.view.setActive(True)
        return False
