from PyQt5 import QtCore, QtWidgets

from util.laser import LaserData, LaserConfig
from gui.qt.tabs import BatchTabs
from gui.qt.tabbeddocks import TabbedDocks
from gui.qt.parameterdlg import ParameterDialog
from gui.qt.laserimage import LaserImageDock

from util.importers import importAgilentBatch
from util.exporters import saveNpz

VERSION = "0.0.1"


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.config = LaserConfig()

        self.setWindowTitle("Laser plot")
        self.resize(1280, 800)

        widget = QtWidgets.QWidget(self)
        self.setCentralWidget(widget)
        layout = QtWidgets.QHBoxLayout()

        self.dockarea = TabbedDocks(self)
        layout.addWidget(self.dockarea, 1)

        widget.setLayout(layout)

        self.createMenus()
        self.statusBar().showMessage("Import or open data to begin.")

        data = importAgilentBatch("/home/tom/Downloads/M1 LUNG 100.b/")
        for n in data.dtype.names:
            w = LaserImageDock(data[n], n, self.config,
                               "/home/tom/Downloads/M1 LUNG 100.b/",
                               self.dockarea)
            w.draw()
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

        # Edit
        config_action = QtWidgets.QAction("&Config", self)
        config_action.setStatusTip("Update the LA-ICP paramaters.")
        config_action.triggered.connect(self.menuConfig)
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
        edit_menu.addAction(config_action)

        view_menu = self.menuBar().addMenu("&View")

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(about_action)

    def menuOpen(self, e):
        paths = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select file(s) to open.", "",
            "Numpy archive (*.npz);;Csv (*.csv)")
        lds = []
        for path in paths:
            if path.lower().endswith('npz'):
                lds += LaserData.open(path)
            else:
                lds.append(importCsv(path))


    def menuSave(self, e):
        path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save file.", "", "Numpy archive (*.npz)")
        lds = [d.data for d in
               self.dockarea.findChildren(QtWidgets.QDockWidget)]
        LaserData.save(path, lds)

    def menuExit(self, e):
        self.close()

    def menuConfig(self, e):
        dlg = ParameterDialog(self, self.config)
        if dlg.exec():
            self.config = dlg.config()
            if dlg.checkAll.checkState() == QtCore.Qt.Checked:
                docks = self.dockarea.findChildren(QtWidgets.QDockWidget)
            else:
                docks = self.dockarea.visibleDocks()
            for d in docks:
                d.config = self.config
                d.draw()

    def menuAbout(self, e):
        QtWidgets.QMessageBox.about(
            self, "About Laser plot",
            ("Visualiser / converter for LA-ICP-MS data.\n"
             f"Version {VERSION}\n"
             "Developed by the UTS Bioimaging Group."))
