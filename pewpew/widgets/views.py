import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets


class ViewSpace(QtWidgets.QSplitter):
    view_index = 0

    def __init__(
        self,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(orientation, parent)
        self.active_view: "View" = None

        self.action_split_horz = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-split-left-right"), "Split &Vertical"
        )
        self.action_split_horz.triggered.connect(self.splitHorizontal)
        self.action_split_vert = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-split-top-bottom"), "Split &Horizontal"
        )
        self.action_split_vert.triggered.connect(self.splitVertical)
        self.action_close_view = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-close"), "Close View"
        )
        self.action_close_view.triggered.connect(self.closeActiveView)

    def closeActiveView(self) -> None:
        if self.active_view is not None:
            self.closeView(self.active_view)
            self.active_view = None

    def closeView(self, view: "View") -> None:
        if view is None:
            return

        if len(self.findChildren(View)) == 1:
            return

        splitter = view.parent()
        if splitter is None:
            return

        view.close()
        view.deleteLater()

        if splitter.count() != 1:
            return

        if splitter != self:
            parent_splitter = splitter.parent()
            if parent_splitter is not None:
                index = parent_splitter.indexOf(splitter)
                sizes = parent_splitter.sizes()
                parent_splitter.insertWidget(index, splitter.widget(0))
                splitter.deleteLater()
                parent_splitter.setSizes(sizes)
        # Doesn't seem to enter here?
        elif isinstance(splitter.widget(0), QtWidgets.QSplitter):
            child_splitter = splitter.widget(0)
            sizes = child_splitter.sizes()
            splitter.setOrientation(child_splitter.orientation())
            splitter.addWidget(child_splitter.widget(0))
            splitter.addWidget(child_splitter.widget(0))
            child_splitter.deleteLater()
            splitter.setSizes(sizes)

    def splitHorizontal(self) -> None:
        self.splitView()

    def splitVertical(self) -> None:
        self.splitView(None, QtCore.Qt.Vertical)

    def splitView(
        self,
        view: QtWidgets.QWidget = None,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
    ) -> None:
        if view is None:
            view = self.activeView()

        splitter = view.parent()
        index = splitter.indexOf(view)

        if splitter.count() == 1:
            size = (splitter.sizes()[0] - splitter.handleWidth()) / 2.0
            splitter.setOrientation(orientation)
            splitter.insertWidget(index - 1, View(self, splitter))
            splitter.setSizes([size, size])
        else:
            sizes = splitter.sizes()
            new_splitter = QtWidgets.QSplitter(orientation)
            new_splitter.setChildrenCollapsible(False)
            new_splitter.addWidget(view)
            new_view = View(self, new_splitter)
            new_splitter.addWidget(new_view)
            splitter.insertWidget(index, new_splitter)

            splitter.setSizes(sizes)

            new_size = (sum(new_splitter.sizes()) - new_splitter.handleWidth()) / 2.0
            new_splitter.setSizes([new_size, new_size])
            self.setActiveView(new_view)

    def activeView(self) -> "View":
        if self.active_view is None:
            view = self.findChildren(View)[0]
            self.active_view = view

        return self.active_view

    def setActiveView(self, view: "View") -> None:
        if self.active_view == view:
            return
        if self.active_view is not None:
            self.active_view.active = False
        view.active = True
        self.active_view = view


class View(QtWidgets.QWidget):
    def __init__(self, viewspace: ViewSpace, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.active = False
        self.viewspace = viewspace

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setFrameStyle(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Sunken)

        self.tabs = ViewTabBar(self.stack, self)
        self.tabs.currentChanged.connect(self.setActive)
        self.titlebar = ViewTitleBar(self.tabs, self)

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.titlebar, 0)
        layout.addWidget(self.stack, 1)
        self.setLayout(layout)

    def setActive(self) -> None:
        self.viewspace.setActiveView(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj and event.type() == QtCore.QEvent.MouseButtonPress:
            self.setActive()
        return False

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            self.tabs.dragEnterEvent(event)
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            self.tabs.dropEvent(event)
        else:
            super().dragEnterEvent(event)


class ViewTabBar(QtWidgets.QTabBar):
    def __init__(
        self, stack: QtWidgets.QStackedWidget, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.stack = stack
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.setElideMode(QtCore.Qt.ElideRight)
        self.setExpanding(False)
        self.setTabsClosable(True)
        # self.setMovable(True)

        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        self.tabCloseRequested.connect(self.removeTab)
        self.currentChanged.connect(self.stack.setCurrentIndex)
        self.tabMoved.connect(self.moveStackWidget)

    def addTab(self, text: str, widget: QtWidgets.QWidget) -> int:
        index = super().addTab(text)
        self.stack.insertWidget(index, widget)
        return index

    def insertTab(self, index: int, text: str, widget: QtWidgets.QWidget) -> int:
        index = super().insertTab(index, text)
        self.stack.insertWidget(index, widget)
        return index

    def tabRemoved(self, index: int) -> None:
        super().tabRemoved(index)
        self.stack.removeWidget(self.stack.widget(index))

    def moveStackWidget(self, ifrom: int, ito: int) -> None:
        self.stack.insertWidget(ito, self.stack.widget(ifrom))

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> QtWidgets.QDialog:
        if event.buttons() == QtCore.Qt.LeftButton:
            index = self.tabAt(event.pos())
            if index == -1:
                return
            dlg = QtWidgets.QInputDialog(self)
            dlg.setWindowTitle("Rename")
            dlg.setLabelText("Name:")
            dlg.setTextValue(self.tabText(index))
            dlg.setInputMode(QtWidgets.QInputDialog.TextInput)
            dlg.textValueSelected.connect(lambda s: self.setTabText(index, s))
            dlg.open()
            return dlg
        else:
            super().mouseDoubleClickEvent(event)
            return None

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() == QtCore.Qt.LeftButton:
            index = self.tabAt(event.pos())

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
                QtGui.QCursor(QtCore.Qt.OpenHandCursor).pixmap(), QtCore.Qt.MoveAction
            )
            drag.start(QtCore.Qt.MoveAction)
        else:
            super().mouseMoveEvent(event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        dest = self.tabAt(event.pos())
        src, ok = event.mimeData().data("application/x-pewpewtabbar").toInt()
        if ok and event.source() == self:
            self.moveTab(src, dest)
        elif ok and isinstance(event.source(), ViewTabBar):
            text = event.source().tabText(src)
            widget = event.source().stack.widget(src)
            event.source().removeTab(src)
            index = self.insertTab(dest, text, widget)
            self.setCurrentIndex(index)
        else:
            super().dropEvent(event)


class ViewTitleBar(QtWidgets.QWidget):
    def __init__(self, tabs: ViewTabBar, view: View):
        super().__init__(view)
        self.view = view

        self.split_button = QtWidgets.QToolButton()
        self.split_button.setAutoRaise(True)
        self.split_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.split_button.setIcon(QtGui.QIcon.fromTheme("view-split-left-right"))
        self.split_button.addAction(self.view.viewspace.action_split_horz)
        self.split_button.addAction(self.view.viewspace.action_split_vert)
        self.split_button.addAction(self.view.viewspace.action_close_view)
        self.split_button.installEventFilter(self.view)

        # Layout the windgets
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        layout.addWidget(line)
        layout.addWidget(tabs, 1)
        layout.addWidget(self.split_button)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        layout.addWidget(line)
        self.setLayout(layout)


if __name__ == "__main__":
    import sys

    sys.path.append("/home/tom/Documents/python/pewpew")
    from laserlib.laser import Laser
    from pewpew.widgets.laser import LaserWidget
    from pewpew.lib.viewoptions import ViewOptions

    app = QtWidgets.QApplication()
    mw = QtWidgets.QMainWindow()
    w = QtWidgets.QWidget()
    w.setMinimumSize(800, 600)

    viewspace = ViewSpace()

    lo = QtWidgets.QVBoxLayout()
    lo.addWidget(viewspace)

    w.setLayout(lo)
    mw.setCentralWidget(w)
    view = View(viewspace)
    viewspace.addWidget(view)
    viewoptions = ViewOptions()
    for i in range(0, 5):
        laser = Laser.from_structured(
            np.array(np.random.random((20, 20)), dtype=[("A1", float), ("B2", float)])
        )
        widget = LaserWidget(laser, viewoptions)
        widget.canvas.drawLaser(widget.laser, "A1")
        view.tabs.addTab(str(i), widget)
    mw.show()
    app.exec_()
