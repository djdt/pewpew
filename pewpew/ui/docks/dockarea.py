import os.path

from PyQt5 import QtCore, QtGui, QtWidgets

from pewpew.ui.events import MousePressRedirectFilter
from pewpew.lib import io as ppio

from laserlib import io
from laserlib.laser import Laser
from laserlib.krisskross import KrissKross

from typing import List
from pewpew.ui.docks import LaserImageDock, KrissKrossImageDock


class DockArea(QtWidgets.QMainWindow):

    mouseSelectFinished = QtCore.pyqtSignal("QWidget*")

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.setDockNestingEnabled(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.setAcceptDrops(True)

        self.mouse_select = False
        self.mouse_filter = MousePressRedirectFilter(self)

    def addDockWidgets(
        self,
        docks: List[QtWidgets.QDockWidget],
        area: QtCore.Qt.DockWidgetArea = QtCore.Qt.LeftDockWidgetArea,
    ) -> None:
        if len(docks) == 0:
            return
        # Add a new dock widget
        super().addDockWidget(area, docks[0])
        origin = self.largestDock(self.visibleDocks())
        if origin is not None:
            self.smartSplitDock(origin, docks[0])
        docks[0].draw()
        docks[0].show()
        for dock in docks[1:]:
            super().addDockWidget(area, dock)
            self.smartSplitDock(
                self.largestDock([d for d in docks if not d.visibleRegion().isEmpty()]),
                dock,
            )
            dock.draw()
            dock.show()

    def orderedDocks(
        self, docks: List[QtWidgets.QDockWidget]
    ) -> List[QtWidgets.QDockWidget]:
        """Returns docks sorted by leftmost / topmost."""
        return sorted(
            docks,
            key=lambda x: (x.geometry().topLeft().x(), x.geometry().topLeft().y()),
        )

    def largestDock(self, docks: List[QtWidgets.QDockWidget]) -> QtWidgets.QDockWidget:
        largest = 0
        dock = None
        for d in self.orderedDocks(docks):
            size = d.size()
            if size.width() > largest:
                largest = size.width()
                dock = d
            if size.height() > largest:
                largest = size.height()
                dock = d
        return dock

    def smartSplitDock(
        self, first: QtWidgets.QDockWidget, second: QtWidgets.QDockWidget
    ) -> None:
        size = first.size()
        minsize = second.minimumSizeHint()
        if size.width() > size.height() and size.width() > 2 * minsize.width():
            self.splitDockWidget(first, second, QtCore.Qt.Horizontal)
        elif size.height() > 2 * minsize.height():
            self.splitDockWidget(first, second, QtCore.Qt.Vertical)
        elif first != second:
            # Split only if there is enough space
            self.tabifyDockWidget(first, second)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("text/uri-list"):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        lasers = []
        csv_as = None
        for url in urls:
            try:
                if url.isLocalFile():
                    path = url.toLocalFile()
                    name, ext = os.path.splitext(path)
                    name = os.path.basename(name)
                    ext = ext.lower()
                    data = None
                    if ext == ".npz":
                        lasers.extend(io.npz.load(path))
                    # TODO we could automate this, no real need for choice
                    elif ext == ".csv":
                        if csv_as is None:
                            choice, ok = QtWidgets.QInputDialog.getItem(
                                self,
                                "Select CSV Type",
                                "CSV Format",
                                ["PewPew", "Raw", "Thermo iCap"],
                                editable=False,
                            )
                            if not ok:
                                return
                            csv_as = choice

                            if csv_as == "PewPew":
                                lasers.append(ppio.csv.load(path))
                            elif csv_as == "Thermo iCap":
                                data = io.thermo.load(path)
                            else:  # Raw
                                data = io.csv.load(path)
                    elif ext == ".txt":
                        data = io.csv.load(path)
                    elif ext == ".b":
                        data = io.agilent.load(path)

                    if data is not None:
                        lasers.append(
                            Laser.from_structured(
                                data=data,
                                config=self.window().config,
                                name=name,
                                filepath=path,
                            )
                        )

            except io.error.LaserLibException as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Import Failed",
                    f"Could not import {os.path.basename(path)}.\n{e}",
                )
                return

        docks = []
        for laser in lasers:
            if isinstance(laser, KrissKross):
                docks.append(KrissKrossImageDock(laser, self))
            else:
                docks.append(LaserImageDock(laser, self))
        self.addDockWidgets(docks)

    def mousePressEvent(self, event: QtCore.QEvent) -> None:
        super().mousePressEvent(event)
        if self.mouse_select is True:
            widget = None
            for dock in self.findChildren(QtWidgets.QDockWidget):
                if dock.underMouse():
                    widget = dock
                    break
            self.mouseSelectFinished.emit(widget)
            self.endMouseSelect()

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        super().keyPressEvent(event)
        if self.mouse_select is True:
            if event.key() == QtCore.Qt.Key_Escape:
                self.mouseSelectFinished.emit(None)
                self.endMouseSelect()

    def startMouseSelect(self) -> None:
        self.mouse_select = True
        for dock in self.findChildren(QtWidgets.QDockWidget):
            if hasattr(dock, "canvas"):
                dock.canvas.installEventFilter(self.mouse_filter)

    def endMouseSelect(self) -> None:
        self.mouse_select = False
        for dock in self.findChildren(QtWidgets.QDockWidget):
            if hasattr(dock, "canvas"):
                dock.canvas.removeEventFilter(self.mouse_filter)

    def tabifyAll(
        self, area: QtCore.Qt.DockWidgetArea = QtCore.Qt.LeftDockWidgetArea
    ) -> None:
        docks = self.findChildren(QtWidgets.QDockWidget)
        # self.removeDockWidget(docks[0])
        self.addDockWidget(docks[0], area)

        for d in docks[1:]:
            self.tabifyDockWidget(docks[0], d)
            d.layout().invalidate()

    def visibleDocks(
        self, dock_type: type = QtWidgets.QDockWidget
    ) -> List[QtWidgets.QDockWidget]:
        docks = self.findChildren(dock_type)
        return [d for d in docks if not d.visibleRegion().isEmpty()]
