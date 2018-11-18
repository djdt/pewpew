from PyQt5 import QtCore, QtWidgets

from util.laser import LaserParams
# from gui.qt.tabs import BatchTabs
from gui.qt.tabbeddocks import TabbedDocks
from gui.qt.controls import Controls
from gui.qt.laserimage import LaserImageDock

from util.importers import importAgilentBatch

VERSION = "0.0.1"


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.params = LaserParams()

        self.setWindowTitle("Laser plot")
        self.resize(1280, 800)

        widget = QtWidgets.QWidget(self)
        self.setCentralWidget(widget)
        layout = QtWidgets.QHBoxLayout()

        self.dockarea = TabbedDocks(self)
        layout.addWidget(self.dockarea)

        self.controls = Controls(self)
        layout.addWidget(self.controls)

        widget.setLayout(layout)

        self.createMenus()
        self.statusBar().showMessage("Import or open data to begin.")

        data = importAgilentBatch("/home/tom/Downloads/HER2 overnight.b")
        for n in data.dtype.names:
            w = LaserImageDock(n, self.dockarea)
            w.drawImage(data[n], self.params, n)
            self.dockarea.addDockWidget(w)


    def createMenus(self):
        # Actions
        # File
        open_action = QtWidgets.QAction("&Open", self)
        open_action.setStatusTip("Open LA-ICP-MS data.")
        open_action.triggered.connect(self.menuOpen)

        save_action = QtWidgets.QAction("&Save", self)
        save_action.setStatusTip("Save to specified format.")
        save_action.triggered.connect(self.menuSave)

        exit_action = QtWidgets.QAction("E&xit", self)
        exit_action.setStatusTip("Quit the program.")
        exit_action.triggered.connect(self.menuExit)

        # View
        # Help
        about_action = QtWidgets.QAction("&About", self)
        about_action.setStatusTip("Import LA-ICP-MS data.")
        about_action.triggered.connect(self.menuAbout)

        # Menus
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        edit_menu = self.menuBar().addMenu("&Edit")

        view_menu = self.menuBar().addMenu("&View")

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(about_action)

    def menuOpen(self, e):
        pass

    def menuSave(self, e):
        pass

    def menuExit(self, e):
        self.close()

    def menuAbout(self, e):
        QtWidgets.QMessageBox.about(
            self, "About Laser plot",
            ("Visualiser / converter for LA-ICP-MS data.\n"
             f"Version {VERSION}\n"
             "Developed by the UTS Bioimaging Group."))
