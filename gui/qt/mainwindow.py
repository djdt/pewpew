from PyQt5 import QtCore, QtGui, QtWidgets

from gui.qt.tabbeddocks import TabbedDocks
from gui.qt.dialogs import ConfigDialog, ColorRangeDialog, TrimDialog
from gui.qt.krisskrosswizard import KrissKrossWizard
from gui.qt.imagedock import ImageDock, LaserImageDock, KrissKrossImageDock

from util.laser import LaserData
from util.krisskross import KrissKrossData
from util.importer import importNpz, importAgilentBatch, importThermoiCapCSV
from util.exporter import exportNpz

import os.path
import traceback

VERSION = "0.2.0"
# TODO implement a smart way to open docks
# check height / width and number to open, can we split them and not violate
# minimum size?

COLORMAPS = [('Perceptually uniform sequential colormap.',
              ['viridis', 'magma']), ('Sequential colormap.', ['gray', 'hot']),
             ('Diverging colormap.', ['Spectral', 'seismic']),
             ('Miscellaneous colormap.', ['nipy_spectral'])]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Defaults for when applying to multiple images
        self.config = LaserData.DEFAULT_CONFIG
        self.viewconfig = ImageDock.DEFAULT_VIEW_CONFIG

        self.setWindowTitle("Laser plot")
        self.resize(1280, 800)

        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QHBoxLayout()

        self.dockarea = TabbedDocks(self)
        layout.addWidget(self.dockarea)

        widget.setLayout(layout)

        self.createMenus()
        self.statusBar().showMessage("Import or open data to begin.")

    def createMenus(self):
        # File
        menu_file = self.menuBar().addMenu("&File")
        action_open = menu_file.addAction(
            QtGui.QIcon.fromTheme('document-open'), "&Open")
        action_open.setShortcut("Ctrl+O")
        action_open.setStatusTip("Open images.")
        action_open.triggered.connect(self.menuOpen)

        action_save = menu_file.addAction(
            QtGui.QIcon.fromTheme('document-save'), "&Save All")
        action_save.setShortcut("Ctrl+S")
        action_save.setStatusTip("Save all images to an archive.")
        action_save.triggered.connect(self.menuSaveAll)

        # File -> Import
        menu_import = menu_file.addMenu("&Import")
        action_import = menu_import.addAction("&Agilent Batch")
        action_import.setStatusTip("Import Agilent image data (.b).")
        action_import.triggered.connect(self.menuImportAgilent)

        action_import = menu_import.addAction("&Thermo iCap CSV")
        action_import.setStatusTip(
            "Import data exported using the CSV export function.")
        action_import.triggered.connect(self.menuImportThermoiCap)

        action_import = menu_import.addAction("&Kriss Kross...")
        action_import.setStatusTip("Start the Kriss Kross import wizard.")
        action_import.triggered.connect(self.menuImportKrissKross)

        menu_file.addSeparator()

        action_exit = menu_file.addAction(
            QtGui.QIcon.fromTheme('application-exit'), "E&xit")
        action_exit.setStatusTip("Quit the program.")
        action_exit.setShortcut("Ctrl+X")
        action_exit.triggered.connect(self.menuExit)

        # Edit
        menu_edit = self.menuBar().addMenu("&Edit")
        # action_calibration = menu_edit.addAction(
        #     QtGui.QIcon.fromTheme('go-top'), "&Calibration")
        # action_calibration.setStatusTip(
        #     "Update the calibrations for visible images.")
        # action_calibration.triggered.connect(self.menuCalibration)
        action_config = menu_edit.addAction(
            QtGui.QIcon.fromTheme('document-properties'), "&Config")
        action_config.setStatusTip("Update the configs for visible images.")
        action_config.setShortcut("Ctrl+K")
        action_config.triggered.connect(self.menuConfig)

        action_trim = menu_edit.addAction(
            QtGui.QIcon.fromTheme('edit-cut'), "&Trim")
        action_trim.setStatusTip("Update trim for visible images.")
        action_trim.setShortcut("Ctrl+T")
        action_trim.triggered.connect(self.menuTrim)

        # View
        menu_view = self.menuBar().addMenu("&View")
        menu_cmap = menu_view.addMenu("&Colormap")
        menu_cmap.setStatusTip("Colormap of displayed images.")

        # View - colormap
        cmap_group = QtWidgets.QActionGroup(menu_cmap)
        for cmap_type, cmaps in COLORMAPS:
            for cmap in cmaps:
                action = cmap_group.addAction(cmap)
                action.setStatusTip(cmap_type)
                action.setCheckable(True)
                if cmap == self.viewconfig['cmap']:
                    action.setChecked(True)
                menu_cmap.addAction(action)
        cmap_group.triggered.connect(self.menuColormap)
        menu_cmap.addSeparator()
        action_cmap_range = menu_cmap.addAction("Range...")
        action_cmap_range.setStatusTip(
            "Set the minimum and maximum values of the colormap.")
        action_cmap_range.triggered.connect(self.menuColormapRange)

        # View - interpolation
        menu_interp = menu_view.addMenu("&Interpolation")
        menu_interp.setStatusTip("Interpolation of displayed images.")
        interp_group = QtWidgets.QActionGroup(menu_interp)
        for interp in [
                'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
                'spline36', 'gaussian'
        ]:
            action = interp_group.addAction(interp)
            action.setCheckable(True)
            if interp == self.viewconfig['interpolation']:
                action.setChecked(True)
            menu_interp.addAction(action)
        interp_group.triggered.connect(self.menuInterpolation)

        action_fontsize = menu_view.addAction("Fontsize")
        action_fontsize.setStatusTip("Set size of font used in images.")
        action_fontsize.triggered.connect(self.menuFontsize)

        menu_view.addSeparator()

        action_refresh = menu_view.addAction(
            QtGui.QIcon.fromTheme('view-refresh'), "Refresh")
        action_refresh.setStatusTip("Redraw all images.")
        action_refresh.setShortcut("F5")
        action_refresh.triggered.connect(self.menuRefresh)
        # Help
        menu_help = self.menuBar().addMenu("&Help")
        action_about = menu_help.addAction(
            QtGui.QIcon.fromTheme('help-about'), "&About")
        action_about.setStatusTip("About this program.")
        action_about.triggered.connect(self.menuAbout)

    def draw(self, visible_only=False):
        if visible_only:
            docks = self.dockarea.visibleDocks()
        else:
            docks = self.dockarea.findChildren(ImageDock)
        for d in docks:
            d.draw()

    def menuOpen(self):
        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open File(s).", "", "Numpy Archives(*.npz);;All files(*)")
        lds = []
        if len(paths) == 0:
            return
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.npz':
                lds += importNpz(path)
            elif ext == '.csv':
                pass
                # lds.append(importCsv(path))
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Open Failed",
                    f"Invalid file type \"{os.path.basename(path)}\".")
        for ld in lds:
            if type(ld) == KrissKrossData:
                dock = KrissKrossImageDock(ld, self.dockarea)
            else:
                dock = LaserImageDock(ld, self.dockarea)
            self.dockarea.addDockWidget(dock)
            dock.draw()

    def menuSaveAll(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", "", "Numpy archives(*.npz);;All files(*)")
        if path == "":
            return
        lds = [d.laser for d in self.dockarea.findChildren(ImageDock)]
        exportNpz(path, lds)

    def menuImportAgilent(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Batch Directory", "")
        if path == "":
            return
        if path.lower().endswith('.b'):
            ld = importAgilentBatch(path, self.config)
            dock = LaserImageDock(ld, self.dockarea)
            self.dockarea.addDockWidget(dock)
            dock.draw()
        else:
            QtWidgets.QMessageBox.warning(
                self, "Import Failed",
                f"Invalid batch directory \"{os.path.basename(path)}\".")

    def menuImportThermoiCap(self):
        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "CSV files(*.csv);;All files(*)", "")

        if len(paths) == 0:
            return
        for path in paths:
            if path.lower().endswith('.csv'):
                ld = importThermoiCapCSV(path, config=self.config)
                dock = LaserImageDock(ld, self.dockarea)
                self.dockarea.addDockWidget(dock)
                dock.draw()
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Import Failed",
                    f"Invalid file format for \"{os.path.basename(path)}\".")

    def menuImportKrissKross(self):
        kkw = KrissKrossWizard(self.config, parent=self)
        if kkw.exec():
            dock = KrissKrossImageDock(kkw.data, self.dockarea)
            self.dockarea.addDockWidget(dock)
            dock.draw()

    def menuExit(self):
        self.close()

    def menuConfig(self):
        dlg = ConfigDialog(self.config, parent=self)
        # Remove the apply button
        dlg.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Ok
                                          | QtWidgets.QDialogButtonBox.Cancel)
        if dlg.exec():
            self.config['spotsize'] = dlg.spotsize
            self.config['speed'] = dlg.speed
            self.config['scantime'] = dlg.scantime
            if dlg.check_all.checkState() == QtCore.Qt.Checked:
                docks = self.dockarea.findChildren(LaserImageDock)
            else:
                docks = self.dockarea.visibleDocks(LaserImageDock)
            for d in docks:
                d.laser.config['spotsize'] = dlg.spotsize
                d.laser.config['speed'] = dlg.speed
                d.laser.config['scantime'] = dlg.scantime
                d.draw()

    def menuTrim(self):
        dlg = TrimDialog([0, 0], parent=self)
        # Remove the apply button
        dlg.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Ok
                                          | QtWidgets.QDialogButtonBox.Cancel)
        if dlg.exec():
            if dlg.check_all.checkState() == QtCore.Qt.Checked:
                docks = self.dockarea.findChildren(LaserImageDock)
            else:
                docks = self.dockarea.visibleDocks(LaserImageDock)
            for d in docks:
                d.laser.setTrim(dlg.trim, dlg.combo_trim.currentText())
                d.draw()

    def menuColormap(self, action):
        self.viewconfig['cmap'] = action.text().replace('&', '')
        self.draw()

    def menuColormapRange(self):
        dlg = ColorRangeDialog(self.viewconfig['cmap_range'], self)
        if dlg.exec():
            cmap_range = dlg.getRangeAsFloatOrPercent()
            self.viewconfig['cmap_range'] = (cmap_range)
            self.draw()

    def menuInterpolation(self, action):
        self.viewconfig['interpolation'] = action.text().replace('&', '')
        self.draw()

    def menuFontsize(self):
        fontsize, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Fontsize",
            "Fontsize",
            value=self.viewconfig['fontsize'],
            min=0,
            max=100,
            step=1)
        if ok:
            self.viewconfig['fontsize'] = fontsize
            self.draw()

    def menuRefresh(self):
        self.draw()

    def menuAbout(self):
        QtWidgets.QMessageBox.about(
            self, "About Laser plot",
            ("Visualiser / converter for LA-ICP-MS data.\n"
             f"Version {VERSION}\n"
             "Developed by the UTS Elemental Bio-Imaging Group."))

    def exceptHook(self, type, value, trace):
        dlg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical,
                                    type.__name__, str(value),
                                    QtWidgets.QMessageBox.NoButton, self)
        dlg.setDetailedText(''.join(traceback.format_tb(trace)))
        tedit = dlg.findChildren(QtWidgets.QTextEdit)
        if tedit[0] is not None:
            tedit[0].setFixedSize(640, 320)
        dlg.exec()

    def closeEvent(self, event):
        for dock in self.dockarea.findChildren(ImageDock):
            dock.close()
        super().closeEvent(event)
