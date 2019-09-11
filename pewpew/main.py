import sys
import copy
import traceback
import os.path

from PySide2 import QtGui, QtWidgets

from laserlib.krisskross import KrissKross

from pewpew import __version__

from pewpew.widgets import dialogs
from pewpew.widgets.exportdialogs import ExportDialog, ExportAllDialog
from pewpew.widgets.prompts import DetailedError
from pewpew.widgets.tools import CalculationsTool, StandardsTool
from pewpew.widgets.wizards import KrissKrossWizard
from pewpew.widgets.laser import LaserViewSpace, LaserWidget

from typing import Callable
from types import TracebackType


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("pew²")
        self.resize(1280, 800)

        self.viewspace = LaserViewSpace()
        self.viewspace.numTabsChanged.connect(self.updateActionAvailablity)
        self.setCentralWidget(self.viewspace)

        self.createActions()
        self.createMenus()
        self.statusBar().showMessage(f"Welcome to pew² version {__version__}.")
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

        self.updateActionAvailablity()

    def createActions(self):
        def qAction(
            icon: str, label: str, status: str, func: Callable
        ) -> QtWidgets.QAction:
            action = QtWidgets.QAction(QtGui.QIcon.fromTheme(icon), label)
            action.setStatusTip(status)
            action.triggered.connect(func)
            return action

        # Laser
        self.action_copy_image = qAction(
            "insert-image",
            "Copy Image",
            "Copy image to clipboard.",
            self.actionCopyImage,
        )
        self.action_config = qAction(
            "document-edit", "Config", "Edit the documents config.", self.actionConfig
        )
        self.action_calibration = qAction(
            "go-top",
            "Calibration",
            "Edit the documents calibration.",
            self.actionCalibration,
        )
        self.action_statistics = qAction(
            "dialog-information",
            "Statistics",
            "Open statisitics dialog for selected data.",
            self.actionStatistics,
        )
        # Laser IO
        self.action_open = qAction(
            "document-open", "&Open", "Open new document(s).", self.actionOpen
        )
        self.action_open.setShortcut("Ctrl+O")
        self.action_save = qAction(
            "document-save", "&Save", "Save document to numpy archive.", self.actionSave
        )
        self.action_export = qAction(
            "document-save-as", "E&xport", "Export documents.", self.actionExport
        )
        # Mainwindow
        self.action_config_default = qAction(
            "document-edit",
            "Default Config",
            "Edit the default config.",
            self.actionConfigDefault,
        )
        self.action_config_default.setShortcut("Ctrl+K")
        self.action_exit = qAction(
            "application-exit", "Quit", "Exit the program.", self.close
        )
        self.action_exit.setShortcut("Ctrl+Shift+Q")
        self.action_toggle_calibrate = qAction(
            "go-top", "Ca&librate", "Toggle calibration.", self.actionToggleCalibrate
        )
        self.action_toggle_calibrate.setShortcut("Ctrl+L")
        self.action_toggle_calibrate.setCheckable(True)
        self.action_toggle_calibrate.setChecked(self.viewspace.options.calibrate)

        self.action_standards_tool = qAction(
            "document-properties",
            "Standards Tool",
            "Open the standards calibration tool.",
            self.actionStandardsTool,
        )
        self.action_calculations_tool = qAction(
            "document-properties",
            "Calculations Tool",
            "Open the calculations tool.",
            self.actionCalculationsTool,
        )

        self.action_group_colormap = QtWidgets.QActionGroup(self)
        for name, cmap in self.viewspace.options.image.COLORMAPS.items():
            action = self.action_group_colormap.addAction(name)
            action.setStatusTip(
                self.viewspace.options.image.COLORMAP_DESCRIPTIONS[name]
            )
            action.setCheckable(True)
            if cmap == self.viewspace.options.image.cmap:
                action.setChecked(True)
        self.action_group_colormap.triggered.connect(self.actionGroupColormap)

        self.action_colormap_range = qAction(
            "", "Set &Range", "Set the range of the colormap.", self.actionColormapRange
        )
        self.action_colormap_range.setShortcut("Ctrl+R")

        self.action_group_interp = QtWidgets.QActionGroup(self)
        for name, interp in self.viewspace.options.image.INTERPOLATIONS.items():
            action = self.action_group_interp.addAction(name)
            action.setCheckable(True)
            if interp == self.viewspace.options.image.interpolation:
                action.setChecked(True)
        self.action_group_interp.triggered.connect(self.actionGroupInterp)

        self.action_fontsize = qAction(
            "insert-text", "Fontsize", "Set size of fonts.", self.actionFontsize
        )

        self.action_toggle_colorbar = qAction(
            "", "Show Colorbar", "Toggle colorbars.", self.actionToggleColorbar
        )
        self.action_toggle_colorbar.setCheckable(True)
        self.action_toggle_colorbar.setChecked(self.viewspace.options.canvas.colorbar)
        self.action_toggle_label = qAction(
            "", "Show Labels", "Toggle element labels.", self.actionToggleLabel
        )
        self.action_toggle_label.setCheckable(True)
        self.action_toggle_label.setChecked(self.viewspace.options.canvas.label)
        self.action_toggle_scalebar = qAction(
            "", "Show Scalebar", "Toggle scalebar.", self.actionToggleScalebar
        )
        self.action_toggle_scalebar.setCheckable(True)
        self.action_toggle_scalebar.setChecked(self.viewspace.options.canvas.scalebar)

        self.action_refresh = qAction(
            "view-refresh", "Refresh", "Redraw documents.", self.refresh
        )
        self.action_refresh.setShortcut("F5")

        self.action_about = qAction(
            "help-about", "&About", "About pew².", self.actionAbout
        )
        # Mainwindow IO
        self.action_import_agilent = qAction(
            "", "Import Agilent", "Import Agilent batches.", self.actionImportAgilent
        )
        self.action_import_thermo = qAction(
            "", "Import Thermo", "Import Thermo iCap CSVs.", self.actionImportThermo
        )
        self.action_import_srr = qAction(
            "",
            "Import Kriss Kross",
            "Open the Kriss-Kross import wizard.",
            self.actionImportSRR,
        )
        self.action_export_all = qAction(
            "document-save-as",
            "E&xport All",
            "Export all open documents.",
            self.actionExportAll,
        )
        self.action_export_all.setShortcut("Ctrl+X")

    def actionCopyImage(self) -> None:
        widget = self.viewspace.activeWidget()
        widget.canvas.copyToClipboard()

    def actionConfig(self) -> QtWidgets.QDialog:
        def applyDialog(dlg: dialogs.ConfigDialog) -> None:
            if dlg.check_all.isChecked():
                self.viewspace.applyConfig(dlg.config)
            else:
                widget.laser.config = copy.copy(dlg.config)
                widget.refresh()

        widget = self.viewspace.activeWidget()
        dlg = dialogs.ConfigDialog(widget.laser.config, parent=self)
        dlg.applyPressed.connect(applyDialog)
        dlg.open()
        return dlg

    def actionCalibration(self) -> QtWidgets.QDialog:
        def applyPress(dlg: dialogs.CalibrationDialog) -> None:
            if dlg.check_all.isChecked():
                self.viewspace.applyCalibration(dlg.calibrations)
            else:
                for iso in widget.laser.isotopes:
                    if iso in dlg.calibrations:
                        widget.laser.data[iso].calibration = copy.copy(
                            dlg.calibrations[iso]
                        )
                widget.refresh()

        widget = self.viewspace.activeWidget()
        calibrations = {
            k: widget.laser.data[k].calibration for k in widget.laser.data.keys()
        }
        dlg = dialogs.CalibrationDialog(
            calibrations, widget.combo_isotopes.currentText(), parent=self
        )
        dlg.applyPressed.connect(self.viewspace.applyCalibration)
        dlg.open()
        return dlg

    def actionStatistics(self) -> QtWidgets.QDialog:
        widget = self.viewspace.activeWidget()
        data = widget.canvas.getMaskedData()
        area = (
            widget.laser.config.get_pixel_width()
            * widget.laser.config.get_pixel_height()
        )
        dlg = dialogs.StatsDialog(
            data, area, widget.canvas.image.get_clim(), parent=self
        )
        dlg.open()
        return dlg

    def actionOpen(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(
            self,
            "Open File(s).",
            "",
            "CSV Documents(*.csv *.txt);;Numpy Archives(*.npz);;"
            "Pew Pew Sessions(*.pew);;All files(*)",
        )
        dlg.selectNameFilter("All files(*)")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.filesSelected.connect(self.viewspace.openDocument)
        dlg.open()
        return dlg

    def actionSave(self) -> QtWidgets.QDialog:
        widget = self.viewspace.activeWidget()
        filepath = widget.laser.filepath
        if filepath.lower().endswith(".npz") and os.path.exists(filepath):
            self.viewspace.saveDocument(filepath)
            return None
        else:
            filepath = widget.laserFilePath()
        dlg = QtWidgets.QFileDialog(
            self, "Save File", filepath, "Numpy archive(*.npz);;All files(*)"
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.fileSelected.connect(self.viewspace.saveDocument)
        dlg.open()
        return dlg

    def actionExport(self) -> QtWidgets.QDialog:
        widget = self.viewspace.activeWidget()
        if widget is None:
            return None
        dlg = ExportDialog(
            widget.laser,
            widget.combo_isotopes.currentText(),
            widget.canvas.view_limits,
            widget.canvas.viewoptions,
            self,
        )
        dlg.open()
        return dlg

    def actionConfigDefault(self) -> QtWidgets.QDialog:
        dlg = dialogs.ConfigDialog(self.viewspace.config, parent=self)
        dlg.check_all.setChecked(True)
        dlg.check_all.setEnabled(False)
        dlg.applyPressed.connect(self.viewspace.applyConfig)
        dlg.open()
        return dlg

    def actionToggleCalibrate(self, checked: bool) -> None:
        self.viewspace.options.calibrate = checked
        self.refresh()

    def actionStandardsTool(self) -> QtWidgets.QDialog:
        def applyTool(tool: StandardsTool) -> None:
            self.viewspace.applyCalibration(tool.calibrations)

        widget = self.viewspace.activeWidget()
        tool = StandardsTool(widget, self.viewspace.options, parent=self)
        tool.applyPressed.connect(applyTool)
        tool.mouseSelectStarted.connect(self.viewspace.mouseSelectStart)
        tool.mouseSelectEnded.connect(self.viewspace.mouseSelectEnd)
        tool.show()
        return tool

    def actionCalculationsTool(self) -> QtWidgets.QDialog:
        widget = self.viewspace.activeWidget()
        tool = CalculationsTool(widget, self.viewspace.options, parent=self)
        # tool.applyPressed.connect(self.refresh())
        tool.mouseSelectStarted.connect(self.viewspace.mouseSelectStart)
        tool.mouseSelectEnded.connect(self.viewspace.mouseSelectEnd)
        tool.show()
        return tool

    def actionGroupColormap(self, action: QtWidgets.QAction) -> None:
        text = action.text().replace("&", "")
        self.viewspace.options.image.set_cmap(text)
        self.refresh()

    def actionColormapRange(self) -> QtWidgets.QDialog:
        def applyDialog(dialog: dialogs.ApplyDialog) -> None:
            for isotope, range in dialog.ranges.items():
                self.viewspace.options.colors.set_range(range, isotope)
            self.viewspace.options.colors.default_range = dialog.default_range
            self.refresh()

        dlg = dialogs.ColorRangeDialog(
            self.viewspace.options, self.viewspace.uniqueIsotopes(), parent=self
        )
        dlg.combo_isotopes.currentTextChanged.connect(self.viewspace.setCurrentIsotope)
        dlg.applyPressed.connect(applyDialog)
        dlg.open()
        return dlg

    def actionGroupInterp(self, action: QtWidgets.QAction) -> None:
        text = action.text().replace("&", "")
        self.viewspace.options.image.interpolation = text
        self.refresh()

    def actionFontsize(self) -> None:
        def applyDialog(size: int) -> None:
            self.viewspace.options.font.size = size
            self.refresh()

        dlg = QtWidgets.QInputDialog(self)
        dlg.setWindowTitle("Fontsize")
        dlg.setLabelText("Fontisze:")
        dlg.setIntValue(self.viewspace.options.font.size)
        dlg.setIntRange(0, 100)
        dlg.setInputMode(QtWidgets.QInputDialog.IntInput)
        dlg.intValueSelected.connect(applyDialog)
        dlg.open()
        return dlg

    def actionToggleColorbar(self, checked: bool) -> None:
        self.viewspace.options.canvas.colorbar = checked
        # Hard refresh
        for view in self.viewspace.views:
            for widget in view.widgets():
                view_limits = widget.canvas.view_limits
                widget.canvas.redrawFigure()
                widget.canvas.view_limits = view_limits
                widget.draw()

    def actionToggleLabel(self, checked: bool) -> None:
        self.viewspace.options.canvas.label = checked
        self.refresh()

    def actionToggleScalebar(self, checked: bool) -> None:
        self.viewspace.options.canvas.scalebar = checked
        self.refresh()

    def actionImportAgilent(self) -> QtWidgets.QDialog:
        dlg = dialogs.MultipleDirDialog(self, "Batch Directories", "")
        dlg.filesSelected.connect(self.viewspace.openDocument)
        dlg.open()
        return dlg

    def actionImportThermo(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(
            self, "Import iCAP Data", "", "iCAP CSV Documents(*.csv);;All Files(*)"
        )
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.filesSelected.connect(self.viewspace.openDocument)
        dlg.open()
        return dlg

    def actionImportSRR(self) -> QtWidgets.QWizard:
        def wizardComplete(laser: KrissKross) -> None:
            view = self.viewspace.activeView()
            view.addTab(laser.name, LaserWidget(laser, self.viewspace.options))

        wiz = KrissKrossWizard(config=self.viewspace.config, parent=self)
        wiz.laserImported.connect(wizardComplete)
        wiz.open()
        return wiz

    def actionExportAll(self) -> QtWidgets.QDialog:
        lasers = []
        for view in self.viewspace.views:
            lasers.extend(w.laser for w in view.widgets())
        dlg = ExportAllDialog(
            lasers, self.viewspace.uniqueIsotopes(), self.viewspace.options, self
        )
        dlg.open()
        return dlg

    def actionAbout(self) -> QtWidgets.QDialog:
        QtWidgets.QMessageBox.about(
            self,
            "About pew²",
            (
                "Visualiser / converter for LA-ICP-MS data.\n"
                f"Version {__version__}\n"
                "Developed by the Atomic Medicine Initiative.\n"
                "https://github.com/djdt/pewpew"
            ),
        )

    def createMenus(self) -> None:
        # File
        menu_file = self.menuBar().addMenu("&File")
        menu_file.addAction(self.action_open)
        # File -> Import
        menu_import = menu_file.addMenu("&Import")
        menu_import.addAction(self.action_import_agilent)
        menu_import.addAction(self.action_import_thermo)
        menu_import.addAction(self.action_import_srr)
        menu_file.addSeparator()

        menu_file.addAction(self.action_export_all)
        menu_file.addSeparator()

        menu_file.addAction(self.action_exit)

        # Edit
        menu_edit = self.menuBar().addMenu("&Edit")
        menu_edit.addAction(self.action_config_default)
        menu_edit.addAction(self.action_toggle_calibrate)
        menu_edit.addSeparator()

        menu_edit.addAction(self.action_standards_tool)
        menu_edit.addAction(self.action_calculations_tool)

        # View
        menu_view = self.menuBar().addMenu("&View")
        menu_cmap = menu_view.addMenu("&Colormap")
        menu_cmap.setStatusTip("Colormap of displayed images.")
        menu_cmap.addActions(self.action_group_colormap.actions())
        menu_cmap.addAction(self.action_colormap_range)

        # View - interpolation
        menu_interp = menu_view.addMenu("&Interpolation")
        menu_interp.setStatusTip("Interpolation of displayed images.")
        menu_interp.addActions(self.action_group_interp.actions())

        menu_view.addAction(self.action_fontsize)
        menu_view.addSeparator()

        menu_view.addAction(self.action_toggle_colorbar)
        menu_view.addAction(self.action_toggle_label)
        menu_view.addAction(self.action_toggle_scalebar)
        menu_view.addSeparator()

        menu_view.addAction(self.action_refresh)

        # Help
        menu_help = self.menuBar().addMenu("&Help")
        menu_help.addAction(self.action_about)

    def buttonStatusUnit(self, toggled: bool) -> None:
        if self.button_status_um.isChecked():
            self.viewspace.options.units = "μm"
        elif self.button_status_row.isChecked():
            self.viewspace.options.units = "row"
        else:  # seconds
            self.viewspace.options.units = "second"

    def refresh(self) -> None:
        self.viewspace.refresh()

    def updateActionAvailablity(self) -> None:
        enabled = self.viewspace.countViewTabs() > 0
        self.action_copy_image.setEnabled(enabled)
        self.action_save.setEnabled(enabled)
        self.action_config.setEnabled(enabled)
        self.action_calibration.setEnabled(enabled)
        self.action_statistics.setEnabled(enabled)
        self.action_export.setEnabled(enabled)
        self.action_export_all.setEnabled(enabled)
        self.action_standards_tool.setEnabled(enabled)
        self.action_calculations_tool.setEnabled(enabled)

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
