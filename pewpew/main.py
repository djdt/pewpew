import sys
import traceback
import copy

from PySide2 import QtCore, QtGui, QtWidgets

from laserlib import LaserConfig
from laserlib.io.error import LaserLibException
from laserlib.krisskross import KrissKross

from pewpew import __version__

from pewpew.lib import io
from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets import dialogs
from pewpew.widgets.exportdialogs import ExportAllDialog
from pewpew.widgets.prompts import DetailedError
from pewpew.widgets.tools import Tool, CalculationsTool, StandardsTool
from pewpew.widgets.wizards import KrissKrossWizard
from pewpew.widgets.laser import LaserViewSpace

from types import TracebackType


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        # Defaults for when applying to multiple images
        self.config = LaserConfig()
        self.viewoptions = ViewOptions()
        self.setWindowTitle("PewPew")
        self.resize(1280, 800)

        self.viewspace = LaserViewSpace()

        # widget = QtWidgets.QWidget()
        # layout = QtWidgets.QVBoxLayout()
        # layout.addWidget(self.viewspace)
        # widget.setLayout(layout)
        self.setCentralWidget(self.viewspace)
        # self.dockarea.numberDocksChanged.connect(self.docksAddedOrRemoved)

        self.createActions()
        self.createMenus()
        self.statusBar().showMessage(f"Welcome to PewPew version {__version__}.")
        self.button_status_um = QtWidgets.QRadioButton("μ")
        self.button_status_row = QtWidgets.QRadioButton("r")
        self.button_status_s = QtWidgets.QRadioButton("s")
        self.button_status_um.setChecked(True)
        self.button_status_um.toggled.connect(self.buttonStatusUnit)
        self.button_status_row.toggled.connect(self.buttonStatusUnit)
        self.button_status_s.toggled.connect(self.buttonStatusUnit)
        self.statusBar().addPermanentWidget(self.button_status_um)
        self.statusBar().addPermanentWidget(self.button_status_row)
        self.statusBar().addPermanentWidget(self.button_status_s)

    def createActions(self):
        self.action_open = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("document-open"), "&Open"
        )
        self.action_open.setShortcut("Save document to numpy archive.")
        self.action_open.setStatusTip("Open new documents.")
        self.action_open.triggered.connect(self.actionOpen)

    def actionOpen(self) -> None:
        view = self.viewspace.activeView()
        dlg = QtWidgets.QFileDialog(
            self,
            "Open File(s).",
            "",
            "CSV Documents(*.csv *.txt);;Numpy Archives(*.npz);;"
            "Pew Pew Sessions(*.pew);;All files(*)",
        )
        dlg.selectNameFilter("All files(*)")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.filesSelected.connect(view.openDocument)
        dlg.open()
        return dlg

    def createMenus(self) -> None:
        # File
        menu_file = self.menuBar().addMenu("&File")
        menu_file.addAction(self.action_open)
        # action_open = menu_file.addAction(
        #     QtGui.QIcon.fromTheme("document-open"), "&Open"
        # )
        # action_open.setShortcut("Ctrl+O")
        # action_open.setStatusTip("Open sessions and images.")
        # action_open.triggered.connect(self.menuOpen)

        # File -> Import
        menu_import = menu_file.addMenu("&Import")
        action_import = menu_import.addAction("&Agilent Batch")
        action_import.setStatusTip("Import Agilent image data (.b).")
        action_import.triggered.connect(self.menuImportAgilent)

        action_import = menu_import.addAction("&Thermo iCap CSV")
        action_import.setStatusTip(
            "Import data exported using the CSV export function."
        )
        action_import.triggered.connect(self.menuImportThermoiCap)

        action_import = menu_import.addAction("&Kriss Kross...")
        action_import.setStatusTip("Start the Kriss Kross import wizard.")
        action_import.triggered.connect(self.menuImportKrissKross)

        menu_file.addSeparator()

        action_save = menu_file.addAction(
            QtGui.QIcon.fromTheme("document-save"), "&Save Session"
        )
        action_save.setShortcut("Ctrl+S")
        action_save.setStatusTip("Save session to file.")
        action_save.triggered.connect(self.menuSaveSession)
        action_save.setEnabled(False)

        self.action_export = menu_file.addAction(
            QtGui.QIcon.fromTheme("document-save-as"), "&Export all"
        )
        self.action_export.setShortcut("Ctrl+X")
        self.action_export.setStatusTip("Export all images to a different format.")
        self.action_export.triggered.connect(self.menuExportAll)
        self.action_export.setEnabled(False)

        menu_file.addSeparator()

        action_close = menu_file.addAction(
            QtGui.QIcon.fromTheme("edit-delete"), "Close All"
        )
        action_close.setStatusTip("Close all open images.")
        action_close.setShortcut("Ctrl+Q")
        action_close.triggered.connect(self.menuCloseAll)

        menu_file.addSeparator()

        action_exit = menu_file.addAction(
            QtGui.QIcon.fromTheme("application-exit"), "E&xit"
        )
        action_exit.setStatusTip("Quit the program.")
        action_exit.setShortcut("Ctrl+Shift+Q")
        action_exit.triggered.connect(self.menuExit)

        # Edit
        menu_edit = self.menuBar().addMenu("&Edit")
        action_config = menu_edit.addAction(
            QtGui.QIcon.fromTheme("document-edit"), "&Config"
        )
        action_config.setStatusTip("Update the configs for visible images.")
        action_config.setShortcut("Ctrl+K")
        action_config.triggered.connect(self.menuConfig)

        action_calibrate = menu_edit.addAction("Ca&librate")
        action_calibrate.setStatusTip("Toggle calibration.")
        action_calibrate.setShortcut("Ctrl+L")
        action_calibrate.setCheckable(True)
        action_calibrate.setChecked(True)
        action_calibrate.toggled.connect(self.menuCalibrate)

        menu_edit.addSeparator()

        self.action_calibration = menu_edit.addAction(
            QtGui.QIcon.fromTheme("document-properties"), "&Standards"
        )
        self.action_calibration.setStatusTip(
            "Generate calibration curve from a standard."
        )
        self.action_calibration.triggered.connect(self.menuStandardsTool)
        self.action_calibration.setEnabled(False)

        self.action_operations = menu_edit.addAction(
            QtGui.QIcon.fromTheme("document-properties"), "&Operations"
        )
        self.action_operations.setStatusTip("Perform calculations using laser data.")
        self.action_operations.triggered.connect(self.menuOperationsTool)
        self.action_operations.setEnabled(False)

        # View
        menu_view = self.menuBar().addMenu("&View")
        menu_cmap = menu_view.addMenu("&Colormap")
        menu_cmap.setStatusTip("Colormap of displayed images.")

        # View - colormap
        cmap_group = QtWidgets.QActionGroup(menu_cmap)
        for name, cmap in self.viewoptions.image.COLORMAPS.items():
            action = cmap_group.addAction(name)
            action.setStatusTip(self.viewoptions.image.COLORMAP_DESCRIPTIONS[name])
            action.setCheckable(True)
            if cmap == self.viewoptions.image.cmap:
                action.setChecked(True)
            menu_cmap.addAction(action)
        cmap_group.triggered.connect(self.menuColormap)
        menu_cmap.addSeparator()
        action_cmap_range = menu_cmap.addAction("&Range...")
        action_cmap_range.setStatusTip(
            "Set the minimum and maximum values or percentiles of the colormap."
        )
        action_cmap_range.setShortcut("Ctrl+R")
        action_cmap_range.triggered.connect(self.menuColormapRange)

        # View - interpolation
        menu_interp = menu_view.addMenu("&Interpolation")
        menu_interp.setStatusTip("Interpolation of displayed images.")
        interp_group = QtWidgets.QActionGroup(menu_interp)
        for name, interp in self.viewoptions.image.INTERPOLATIONS.items():
            action = interp_group.addAction(name)
            action.setCheckable(True)
            if interp == self.viewoptions.image.interpolation:
                action.setChecked(True)
            menu_interp.addAction(action)
        interp_group.triggered.connect(self.menuInterpolation)

        action_fontsize = menu_view.addAction("Fontsize")
        action_fontsize.setStatusTip("Set size of font used in images.")
        action_fontsize.triggered.connect(self.menuFontsize)

        menu_view.addSeparator()

        action_option_colorbar = menu_view.addAction("Show Colorbars")
        action_option_colorbar.setCheckable(True)
        action_option_colorbar.setChecked(True)
        action_option_colorbar.toggled.connect(self.menuOptionColorbar)

        action_option_colorbar = menu_view.addAction("Show Labels")
        action_option_colorbar.setCheckable(True)
        action_option_colorbar.setChecked(True)
        action_option_colorbar.toggled.connect(self.menuOptionLabel)

        action_option_colorbar = menu_view.addAction("Show Scalebars")
        action_option_colorbar.setCheckable(True)
        action_option_colorbar.setChecked(True)
        action_option_colorbar.toggled.connect(self.menuOptionScalebar)

        menu_view.addSeparator()

        action_refresh = menu_view.addAction(
            QtGui.QIcon.fromTheme("view-refresh"), "Refresh"
        )
        action_refresh.setStatusTip("Redraw all images.")
        action_refresh.setShortcut("F5")
        action_refresh.triggered.connect(self.menuRefresh)

        # Help
        menu_help = self.menuBar().addMenu("&Help")
        action_about = menu_help.addAction(
            QtGui.QIcon.fromTheme("help-about"), "&About"
        )
        action_about.setStatusTip("About this program.")
        action_about.triggered.connect(self.menuAbout)

    def buttonStatusUnit(self, toggled: bool) -> None:
        if self.button_status_um.isChecked():
            self.viewoptions.units = "μm"
        elif self.button_status_row.isChecked():
            self.viewoptions.units = "row"
        else:  # seconds
            self.viewoptions.units = "second"

    def refresh(self) -> None:
        self.viewspace.refresh()

    def updateActionAvailablity(self) -> None:
        enabled = self.viewspace.countViewTabs() > 0
        self.action_export.setEnabled(enabled)
        self.action_calibration.setEnabled(enabled)
        self.action_operations.setEnabled(enabled)

    def menuOpen(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(
            self,
            "Open File(s).",
            "",
            "CSV Documents(*.csv *.txt);;Numpy Archives(*.npz);;"
            "Pew Pew Sessions(*.pew);;All files(*)",
        )
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.filesSelected.connect(self.importFiles)
        dlg.open()
        return dlg

    def importFiles(self, paths: str) -> None:
        try:
            lasers = io.import_any(paths, self.config)
        except LaserLibException as e:
            QtWidgets.QMessageBox.critical(self, type(e).__name__, f"{e}")
            return

        docks = []
        for laser in lasers:
            if isinstance(laser, KrissKross):
                docks.append(KrissKrossImageDock(laser, self.viewoptions))
            else:
                docks.append(LaserImageDock(laser, self.viewoptions))
        self.dockarea.addDockWidgets(docks)

    def menuImportAgilent(self) -> QtWidgets.QDialog:
        dlg = dialogs.MultipleDirDialog(self, "Batch Directories", "")
        dlg.filesSelected.connect(self.importFiles)
        dlg.open()
        return dlg

    def menuImportThermoiCap(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(
            self, "Import iCAP Data", "", "iCAP CSV Documents(*.csv);;All Files(*)"
        )
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.filesSelected.connect(self.importFiles)
        dlg.open()
        return dlg

    def menuImportKrissKross(self) -> QtWidgets.QWizard:
        def wizardComplete(laser: KrissKross) -> None:
            self.dockarea.addDockWidgets([KrissKrossImageDock(laser, self.viewoptions)])

        wiz = KrissKrossWizard(config=self.config, parent=self)
        wiz.laserImported.connect(wizardComplete)
        wiz.open()
        return wiz

    def menuSaveSession(self) -> None:
        # Save the window state
        settings = QtCore.QSettings("Pewpew", "Pewpew")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

        docks = self.dockarea.findChildren(LaserImageDock)
        if len(docks) == 0:
            QtWidgets.QMessageBox.information(self, "Save Session", "Nothing to save.")
            return
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", "", "Pew Pew sessions(*.pew);;All files(*)"
        )
        if path == "":
            return
        lds = [d.laser for d in docks]
        io.npz.save(path, lds)

    def menuExportAll(self) -> QtWidgets.QDialog:
        docks = self.dockarea.findChildren(LaserImageDock)
        if len(docks) == 0:
            return

        dlg = ExportAllDialog(
            [dock.laser for dock in docks],
            self.dockarea.uniqueIsotopes(),
            self.viewoptions,
            self,
        )
        dlg.open()
        return dlg

    def menuCloseAll(self) -> None:
        for dock in self.dockarea.findChildren(LaserImageDock):
            dock.close()

    def menuExit(self) -> None:
        self.close()

    def menuConfig(self) -> QtWidgets.QDialog:
        def applyDialog(dialog: dialogs.ApplyDialog) -> None:
            self.config = dialog.config
            for dock in self.dockarea.findChildren(LaserImageDock):
                dock.laser.config.spotsize = self.config.spotsize
                dock.laser.config.speed = self.config.speed
                dock.laser.config.scantime = self.config.scantime
                dock.draw()

        dlg = dialogs.ConfigDialog(self.config, parent=self)
        dlg.applyPressed.connect(applyDialog)
        dlg.check_all.setEnabled(False)
        dlg.check_all.setChecked(True)
        dlg.open()
        return dlg

    def menuCalibrate(self, checked: bool) -> None:
        self.viewoptions.calibrate = checked
        self.refresh()

    def menuStandardsTool(self) -> QtWidgets.QDialog:
        def applyTool(tool: Tool) -> None:
            for dock in self.dockarea.findChildren(LaserImageDock):
                for isotope in tool.calibrations.keys():
                    if isotope in dock.laser.isotopes:
                        dock.laser.data[isotope].calibration = copy.copy(
                            tool.calibrations[isotope]
                        )
                dock.draw()

        docks = self.dockarea.orderedDocks(self.dockarea.visibleDocks(LaserImageDock))
        tool = StandardsTool(docks[0], self.viewoptions, parent=self)
        tool.applyPressed.connect(applyTool)
        tool.mouseSelectStarted.connect(self.dockarea.startMouseSelect)
        tool.show()
        return tool

    def menuOperationsTool(self) -> QtWidgets.QDialog:
        docks = self.dockarea.orderedDocks(self.dockarea.visibleDocks(LaserImageDock))
        tool = CalculationsTool(docks[0], self.viewoptions, parent=self)
        tool.applyPressed.connect(docks[0].populateComboIsotopes)
        tool.mouseSelectStarted.connect(self.dockarea.startMouseSelect)
        tool.show()
        return tool

    def menuColormap(self, action: QtWidgets.QAction) -> None:
        text = action.text().replace("&", "")
        self.viewoptions.image.set_cmap(text)
        self.refresh()

    def menuColormapRange(self) -> QtWidgets.QDialog:
        def isotopeChanged(text: str) -> None:
            for dock in self.dockarea.findChildren(LaserImageDock):
                if text in dock.laser.isotopes:
                    dock.combo_isotopes.setCurrentText(text)

        def applyDialog(dialog: dialogs.ApplyDialog) -> None:
            for isotope, range in dialog.ranges.items():
                self.viewoptions.colors.set_range(range, isotope)
            self.viewoptions.colors.default_range = dialog.default_range
            self.refresh()

        dlg = dialogs.ColorRangeDialog(
            self.viewoptions, self.dockarea.uniqueIsotopes(), parent=self
        )
        dlg.combo_isotopes.currentTextChanged.connect(isotopeChanged)
        dlg.applyPressed.connect(applyDialog)
        dlg.open()
        return dlg

    def menuInterpolation(self, action: QtWidgets.QAction) -> None:
        self.viewoptions.image.interpolation = action.text().replace("&", "")
        self.refresh()

    def menuFontsize(self) -> None:
        fontsize, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Fontsize",
            "Fontsize",
            value=self.viewoptions.font.size,
            min=0,
            max=100,
            step=1,
        )
        if ok:
            self.viewoptions.font.size = fontsize
            self.refresh()

    def menuOptionColorbar(self, checked: bool) -> None:
        self.viewoptions.canvas.colorbar = checked
        for dock in self.dockarea.findChildren(LaserImageDock):
            view_limits = dock.canvas.view_limits
            dock.canvas.redrawFigure()
            dock.canvas.view_limits = view_limits
            dock.draw()

    def menuOptionLabel(self, checked: bool) -> None:
        self.viewoptions.canvas.label = checked
        for dock in self.dockarea.findChildren(LaserImageDock):
            dock.draw()

    def menuOptionScalebar(self, checked: bool) -> None:
        self.viewoptions.canvas.scalebar = checked
        for dock in self.dockarea.findChildren(LaserImageDock):
            dock.draw()

    def menuRefresh(self) -> None:
        self.refresh()

    def menuAbout(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About Pewpew",
            (
                "Visualiser / converter for LA-ICP-MS data.\n"
                f"Version {__version__}\n"
                "Developed by the UTS Elemental Bio-Imaging Group.\n"
                "https://github.com/djdt/pewpew"
            ),
        )

    def exceptHook(self, type: type, value: BaseException, tb: TracebackType) -> None:
        if type == KeyboardInterrupt:
            print("Keyboard interrupt, exiting.")
            sys.exit(1)
        DetailedError.critical(
            type.__name__,
            str(value),
            "".join(traceback.format_exception(type, value, tb)),
            self,
        )

    # def closeEvent(self, event: QtCore.QEvent) -> None:
    #     for dock in self.dockarea.findChildren(LaserImageDock):
    #         dock.close()
    #     super().closeEvent(event)
