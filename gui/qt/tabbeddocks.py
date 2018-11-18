from PyQt5 import QtCore, QtWidgets


class TabbedDocks(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setDockNestingEnabled(True)

    def addDockWidget(self, dock, area=QtCore.Qt.LeftDockWidgetArea):
        super().addDockWidget(area, dock)
        # Find a child dock in same area to tab
        docks = self.findChildren(QtWidgets.QDockWidget)
        for d in docks:
            if self.dockWidgetArea(d) == area and d != dock:
                self.tabifyDockWidget(d, dock)
                return
