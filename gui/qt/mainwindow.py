from PyQt5 import QtCore, QtGui, QtWidgets

from util.laser import LaserData
from gui.qt.tabbeddocks import TabbedDocks
from gui.qt.configdialog import ConfigDialog
from gui.qt.laserimage import LaserImageDock

from util.importer import importNpz, importCsv, importAgilentBatch
from util.exporter import exportNpz, exportCsv

import os.path

VERSION = "0.0.2"


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.config = LaserData.DEFAULT_CONFIG
        self.viewconfig = {'cmap': 'viridis'}

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

    def createMenus(self):
        # File
        filemenu = self.menuBar().addMenu("&File")
        openaction = filemenu.addAction(
            QtGui.QIcon.fromTheme('document-open'), "&Open")
        openaction.setShortcut("Ctrl+O")
        openaction.setStatusTip("Open LA-ICP-MS data.")
        openaction.triggered.connect(self.menuOpen)

        saveaction = filemenu.addAction(
            QtGui.QIcon.fromTheme('document-save'), "&Save")
        saveaction.setShortcut("Ctrl+S")
        saveaction.setStatusTip("Save to specified format.")
        saveaction.triggered.connect(self.menuSave)

        filemenu.addSeparator()

        # File -> Import
        importmenu = filemenu.addMenu("&Import")
        importaction = importmenu.addAction("Agilent Batch")
        importaction.setStatusTip("Import Agilent data (.b).")
        importaction.triggered.connect(self.menuImportAgilent)

        exitaction = filemenu.addAction(
            QtGui.QIcon.fromTheme('application-exit'), "E&xit")
        exitaction.setStatusTip("Quit the program.")
        exitaction.triggered.connect(self.menuExit)

        # Edit
        editmenu = self.menuBar().addMenu("&Edit")
        configaction = editmenu.addAction(
            QtGui.QIcon.fromTheme('document-properties'), "Config")
        configaction.setStatusTip("Update the LA-ICP paramaters.")
        configaction.triggered.connect(self.menuConfig)
        # View
        viewmenu = self.menuBar().addMenu("&View")
        cmapmenu = viewmenu.addMenu("&Colormap")
        cmapmenu.setStatusTip("Change the image colormap.")
        cmapgroup = QtWidgets.QActionGroup(cmapmenu)
        for cmap in ['magma', 'viridis', 'plasma', 'nipy_spectral',
                     'gnuplot2', 'CMRmap']:
            action = cmapgroup.addAction(cmap)
            action.setCheckable(True)
            if cmap == 'magma':
                action.setChecked(True)
            cmapmenu.addAction(action)
        cmapgroup.triggered.connect(self.menuColormap)
        # Help
        helpmenu = self.menuBar().addMenu("&Help")
        aboutaction = helpmenu.addAction(
            QtGui.QIcon.fromTheme('help-about'), "&About")
        aboutaction.setStatusTip("Import LA-ICP-MS data.")
        aboutaction.triggered.connect(self.menuAbout)

    def menuOpen(self):
        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select file(s) to open.", "", "(*.npz *.csv);;All files(*)")
        lds = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.npz':
                lds += importNpz(path)
            elif ext == '.csv':
                lds.append(importCsv(path))
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Open failed",
                    f"Invalid file type \'{os.path.basename(path)}\'.")
        for ld in lds:
            dock = LaserImageDock(ld, self.dockarea)
            dock.draw(cmap=self.viewconfig['cmap'])
            self.dockarea.addDockWidget(dock)

    def menuSave(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save file.", "", "Numpy Archive(*.npz);;All files(*)")
        lds = [d.laserdata for d in
               self.dockarea.findChildren(QtWidgets.QDockWidget)]
        exportNpz(path, lds)

    def menuImportAgilent(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Batch directory", "")
        if path.endswith('.b'):
            lds = importAgilentBatch(path, self.config)
            for ld in lds:
                dock = LaserImageDock(ld, self.dockarea)
                dock.draw(cmap=self.viewconfig['cmap'])
                self.dockarea.addDockWidget(dock)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Import failed",
                f"Invalid batch directory \'{os.path.basename(path)}\'.")

    def menuExit(self):
        self.close()

    def menuConfig(self):
        dlg = ConfigDialog(self, self.config)
        if dlg.exec():
            self.config = dlg.config
            if dlg.checkAll.checkState() == QtCore.Qt.Checked:
                docks = self.dockarea.findChildren(QtWidgets.QDockWidget)
            else:
                docks = self.dockarea.visibleDocks()
            for d in docks:
                d.laserdata.config = self.config
                d.draw(cmap=self.viewconfig['cmap'])

    def menuColormap(self, action):
        self.viewconfig['cmap'] = action.text().lstrip('&')
        for dock in self.dockarea.findChildren(QtWidgets.QDockWidget):
            dock.draw(cmap=self.viewconfig['cmap'])

    def menuAbout(self):
        QtWidgets.QMessageBox.about(
            self, "About Laser plot",
            ("Visualiser / converter for LA-ICP-MS data.\n"
             f"Version {VERSION}\n"
             "Developed by the UTS Bioimaging Group."))
