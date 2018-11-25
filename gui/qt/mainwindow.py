from PyQt5 import QtCore, QtGui, QtWidgets

from gui.qt.tabbeddocks import TabbedDocks
from gui.qt.configdialog import ConfigDialog
from gui.qt.krisskrosswizard import KrissKrossWizard
from gui.qt.laserimagedock import ImageDock, LaserImageDock, KrissKrossImageDock

from util.laser import LaserData
from util.importer import importNpz, importCsv, importAgilentBatch
from util.exporter import exportNpz

import os.path

VERSION = "0.0.2"


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.config = LaserData.DEFAULT_CONFIG
        self.viewconfig = {'cmap': 'magma', 'cmap_range': (1, 99)}

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

        # config = {'scantime': 0.1, 'spotsize': 10, 'speed': 10, 'gradient': 1.0, 'intercept': 0.0}
        # self.config = config
        # lds = importAgilentBatch("/home/tom/Downloads/raw/Horz.b", config)
        # lds2 = importAgilentBatch("/home/tom/Downloads/raw/Vert.b", config)
        # for ld in lds:
        #     dock = LaserImageDock(ld, self.dockarea)
        #     dock.draw(cmap=self.viewconfig['cmap'])
        #     self.dockarea.addDockWidget(dock)

        # kkd = KrissKrossData(config=config)
        # kkd.fromLayers((lds[1].data, lds2[1].data))
        # dock = KrissKrossImageDock(kkd, self.dockarea)
        # dock.draw(cmap=self.viewconfig['cmap'])
        # self.dockarea.addDockWidget(dock)

    def createMenus(self):
        # File
        file_menu = self.menuBar().addMenu("&File")
        open_action = file_menu.addAction(
            QtGui.QIcon.fromTheme('document-open'), "&Open")
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open LA-ICP-MS data.")
        open_action.triggered.connect(self.menuOpen)

        save_action = file_menu.addAction(
            QtGui.QIcon.fromTheme('document-save'), "&Save")
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("Save to specified format.")
        save_action.triggered.connect(self.menuSave)

        file_menu.addSeparator()

        # File -> Import
        import_menu = file_menu.addMenu("&Import")
        import_action = import_menu.addAction("&Agilent Batch")
        import_action.setStatusTip("Import Agilent data (.b).")
        import_action.triggered.connect(self.menuImportAgilent)

        import_action = import_menu.addAction("&Kriss Kross...")
        import_action.setStatusTip("Start the Kriss Kross import wizard.")
        import_action.triggered.connect(self.menuImportKrissKross)

        exit_action = file_menu.addAction(
            QtGui.QIcon.fromTheme('application-exit'), "E&xit")
        exit_action.setStatusTip("Quit the program.")
        exit_action.setShortcut("Ctrl+X")
        exit_action.triggered.connect(self.menuExit)

        # Edit
        edit_menu = self.menuBar().addMenu("&Edit")
        config_action = edit_menu.addAction(
            QtGui.QIcon.fromTheme('document-properties'), "Config")
        config_action.setStatusTip("Update the LA-ICP paramaters.")
        config_action.triggered.connect(self.menuConfig)
        # View
        view_menu = self.menuBar().addMenu("&View")
        cmap_menu = view_menu.addMenu("&Colormap")
        cmap_menu.setStatusTip("Change the image colormap.")
        cmap_group = QtWidgets.QActionGroup(cmap_menu)
        for cmap in ['magma', 'viridis', 'plasma', 'nipy_spectral',
                     'gnuplot2', 'CMRmap']:
            action = cmap_group.addAction(cmap)
            action.setCheckable(True)
            if cmap == 'magma':
                action.setChecked(True)
            cmap_menu.addAction(action)
        cmap_group.triggered.connect(self.menuColormap)
        # Help
        help_menu = self.menuBar().addMenu("&Help")
        about_action = help_menu.addAction(
            QtGui.QIcon.fromTheme('help-about'), "&About")
        about_action.setStatusTip("Import LA-ICP-MS data.")
        about_action.triggered.connect(self.menuAbout)

    def menuOpen(self):
        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select file(s) to open.", "", "(*.npz *.csv);;All files(*)")
        lds = []
        if len(paths) == 0:
            return
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
        if path == "":
            return
        lds = [d.laserdata for d in
               self.dockarea.findChildren(ImageDock)]
        exportNpz(path, lds)

    def menuImportAgilent(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Batch directory", "")
        if path == "":
            return
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

    def menuImportKrissKross(self):
        kkw = KrissKrossWizard(self.config, self)
        if kkw.exec():
            for kkd in kkw.krisskrossdata:
                print(kkd.data.shape)
                dock = KrissKrossImageDock(kkd, self.dockarea)
                dock.draw(cmap=self.viewconfig['cmap'])
                self.dockarea.addDockWidget(dock)

    def menuExit(self):
        self.close()

    def menuConfig(self):
        dlg = ConfigDialog(self.config, parent=self)
        if dlg.exec():
            self.config = dlg.form.config
            if dlg.checkAll.checkState() == QtCore.Qt.Checked:
                docks = self.dockarea.findChildren(ImageDock)
            else:
                docks = self.dockarea.visibleDocks()
            for d in docks:
                if type(d) == KrissKrossImageDock:
                    d.kkdata.config = self.config
                else:
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
