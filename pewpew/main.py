import os.path
import sys
import traceback
import copy

from PySide2 import QtCore, QtGui, QtWidgets

from laserlib import io
from laserlib import Laser, LaserConfig, LaserData
from laserlib.krisskross import KrissKross, KrissKrossData

from pewpew import __version__

from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets import dialogs
from pewpew.widgets.docks import LaserImageDock, KrissKrossImageDock
from pewpew.widgets.exporters import ExportAllDialog
from pewpew.widgets.prompts import DetailedError, OverwriteFilePrompt
from pewpew.widgets.tools import get_operations_tool, Tool, StandardsTool
from pewpew.widgets.windows import DockArea
from pewpew.widgets.wizards import KrissKrossWizard

from typing import List
from types import TracebackType


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        # Defaults for when applying to multiple images
        self.config = LaserConfig()
        self.viewoptions = ViewOptions()
        self.setWindowTitle("Pew Pew")
        self.resize(1280, 800)

        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QHBoxLayout()

        self.dockarea = DockArea(self)
        self.dockarea.numberDocksChanged.connect(self.docksAddedOrRemoved)
        layout.addWidget(self.dockarea)

        widget.setLayout(layout)

        self.createMenus()
        self.statusBar().showMessage("Import or open data to begin.")
        self.button_status_um = QtWidgets.QRadioButton("μ")
        self.button_status_um.setChecked(True)
        self.button_status_row = QtWidgets.QRadioButton("r")
        self.button_status_s = QtWidgets.QRadioButton("s")
        self.button_status_um.toggled.connect(self.buttonStatusUnit)
        self.button_status_row.toggled.connect(self.buttonStatusUnit)
        self.button_status_s.toggled.connect(self.buttonStatusUnit)
        self.statusBar().addPermanentWidget(self.button_status_um)
        self.statusBar().addPermanentWidget(self.button_status_row)
        self.statusBar().addPermanentWidget(self.button_status_s)

    def createMenus(self) -> None:
        # File
        menu_file = self.menuBar().addMenu("&File")
        action_open = menu_file.addAction(
            QtGui.QIcon.fromTheme("document-open"), "&Open"
        )
        action_open.setShortcut("Ctrl+O")
        action_open.setStatusTip("Open sessions and images.")
        action_open.triggered.connect(self.menuOpen)

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

        action_export = menu_file.addAction(
            QtGui.QIcon.fromTheme("document-save-as"), "&Export all"
        )
        action_export.setShortcut("Ctrl+X")
        action_export.setStatusTip("Export all images to a different format.")
        action_export.triggered.connect(self.menuExportAll)

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
        colors = self.viewoptions.colors
        for name in colors.COLORMAPS:
            action = cmap_group.addAction(name)
            action.setStatusTip(colors.COLORMAP_DESCRIPTIONS[name])
            action.setCheckable(True)
            if colors.COLORMAPS[name] == self.viewoptions.colors.cmap:
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
        for interp in self.viewoptions.image.INTERPOLATIONS:
            action = interp_group.addAction(interp)
            action.setCheckable(True)
            if interp == self.viewoptions.image.interpolation:
                action.setChecked(True)
            menu_interp.addAction(action)
        interp_group.triggered.connect(self.menuInterpolation)

        # View - filtering
        # menu_filter = menu_view.addMenu("&Filtering")
        # menu_filter.setStatusTip("Apply filtering to images.")
        # filter_group = QtWidgets.QActionGroup(menu_filter)
        # for filter in MainWindow.FILTERS:
        #     action = filter_group.addAction(filter)
        #     action.setCheckable(True)
        #     if filter == self.viewconfig["filtering"]["type"]:
        #         action.setChecked(True)
        #     menu_filter.addAction(action)
        # filter_group.triggered.connect(self.menuFiltering)

        # action_filter_properties = menu_filter.addAction("Properties...")
        # action_filter_properties.setStatusTip("Set the properties used by filters.")
        # action_filter_properties.triggered.connect(self.menuFilteringProperties)

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

    def refresh(self, visible_only: bool = False) -> None:
        if visible_only:
            docks = self.dockarea.visibleDocks()
        else:
            docks = self.dockarea.findChildren(LaserImageDock)
        for d in docks:
            d.draw()

    def docksAddedOrRemoved(self) -> None:
        num_docks = len(self.dockarea.findChildren(LaserImageDock))
        enabled = not num_docks == 0
        self.action_calibration.setEnabled(enabled)
        self.action_operations.setEnabled(enabled)

    def menuOpen(self) -> None:
        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open File(s).",
            "",
            "CSV files(*.csv *.txt);;Numpy Archives(*.npz);;"
            "Pew Pew Sessions(*.pew);;All files(*)",
            "All files(*)",
        )
        lasers: List[Laser] = []
        if len(paths) == 0:
            return
        for path in paths:
            name, ext = os.path.splitext(path)
            name = os.path.basename(name)
            ext = ext.lower()
            try:
                if ext == ".npz":
                    lasers += io.npz.load(path)
                elif ext in [".csv", ".txt", ".text"]:
                    laser = Laser.from_structured(
                        io.csv.load(path),
                        config=self.config,
                        name=os.path.basename(os.path.splitext(path)[0]),
                        filepath=path,
                    )
                    lasers.append(laser)
                else:
                    raise io.error.LaserLibException("Invalid file extension.")
            except io.error.LaserLibException as e:
                QtWidgets.QMessageBox.warning(
                    self, type(e).__name__, f"{os.path.basename(path)}: {e}"
                )
                return
        docks = []
        for laser in lasers:
            if isinstance(laser, KrissKross):
                docks.append(KrissKrossImageDock(laser, self.dockarea))
            else:
                docks.append(LaserImageDock(laser, self.viewoptions, self.dockarea))
        self.dockarea.addDockWidgets(docks)

    def menuImportAgilent(self) -> None:
        paths = dialogs.MultipleDirDialog.getExistingDirectories(
            self, "Batch Directories", ""
        )
        docks = []
        for path in paths:
            try:
                if path.lower().endswith(".b"):
                    laser = Laser.from_structured(
                        io.agilent.load(path),
                        config=self.config,
                        name=os.path.basename(os.path.splitext(path)[0]),
                        filepath=path,
                    )
                    docks.append(LaserImageDock(laser, self.viewoptions, self.dockarea))
                else:
                    raise io.error.LaserLibException("Invalid batch directory.")
            except io.error.LaserLibException as e:
                QtWidgets.QMessageBox.warning(
                    self, type(e).__name__, f"{os.path.basename(path)}: {e}"
                )
                return
        self.dockarea.addDockWidgets(docks)

    def menuImportThermoiCap(self) -> None:
        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Import CSVs", "", "CSV files(*.csv);;All files(*)"
        )

        if len(paths) == 0:
            return
        docks = []
        for path in paths:
            try:
                if path.lower().endswith(".csv"):
                    laser = Laser.from_structured(
                        io.thermo.load(path),
                        config=self.config,
                        name=os.path.basename(os.path.splitext(path)[0]),
                        filepath=path,
                    )
                    docks.append(LaserImageDock(laser, self.viewoptions, self.dockarea))
                else:
                    raise io.error.LaserLibException("Invalid file.")
            except io.error.LaserLibException as e:
                QtWidgets.QMessageBox.warning(
                    self, type(e).__name__, f"{os.path.basename(path)}: {e}"
                )
                return
        self.dockarea.addDockWidgets(docks)

    def menuImportKrissKross(self) -> None:
        kkw = KrissKrossWizard(config=self.config, parent=self)
        if kkw.exec():
            dock = KrissKrossImageDock(kkw.data, self.viewoptions, self.dockarea)
            self.dockarea.addDockWidgets([dock])

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

    def menuExportAll(self) -> None:
        docks = self.dockarea.findChildren(LaserImageDock)
        if len(docks) == 0:
            return

        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export To Directory", ""
        )
        if not path:
            return

        isotopes = list(set.union(*[set(dock.laser.isotopes) for dock in docks]))
        dlg = ExportAllDialog(path, docks[0].laser.name, isotopes, 1, self)
        if not dlg.open():
            return

        prompt = OverwriteFilePrompt(parent=self)
        for dock in docks:
            paths = dlg.generate_paths(dock.laser, prompt=prompt)
            ext = ExportAllDialog.FORMATS[dlg.combo_formats.currentText()]

            if ext == ".png":
                old_size = dock.canvas.figure.get_size_inches()
                size = dlg.options.png.imagesize()
                dpi = dock.canvas.figure.get_dpi()
                dock.canvas.figure.set_size_inches(size[0] / dpi, size[1] / dpi)

            for path, isotope, _ in paths:
                if ext == ".csv":
                    kwargs = {"calibrate": self.viewoptions.calibrate}
                    if dlg.options.csv.trimmedChecked():
                        kwargs["extent"] = dock.canvas.view_limits
                    io.csv.save(path, dock.laser.get(isotope, **kwargs))
                elif ext == ".npz":
                    io.npz.save(path, [dock.laser])
                elif ext == ".png":
                    dock.canvas.drawLaser(dock.laser, isotope)
                    dock.canvas.figure.savefig(path, transparent=True, frameon=False)

                elif ext == ".vti":
                    spacing = (
                        dock.laser.config.get_pixel_width(),
                        dock.laser.config.get_pixel_height(),
                        dock.laser.config.spotsize / 2.0,
                    )
                    io.vtk.save(
                        path,
                        dock.laser.get_structured(calibrate=self.viewoptions.calibrate),
                        spacing=spacing,
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Format", f"Unable to export {ext} format."
                    )
                    return

            if ext == ".png":
                dock.canvas.figure.set_size_inches(*old_size)
                dock.canvas.drawLaser(dock.laser, dock.combo_isotope.currentText())
                dock.canvas.draw()

    def menuCloseAll(self) -> None:
        for dock in self.dockarea.findChildren(LaserImageDock):
            dock.close()

    def menuExit(self) -> None:
        self.close()

    def menuConfig(self) -> None:
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

        if dlg.open():
            applyDialog(dlg)

    def menuCalibrate(self, checked: bool) -> None:
        self.viewoptions.calibrate = checked
        self.refresh()

    def menuStandardsTool(self) -> None:
        def applyTool(tool: Tool) -> None:
            for dock in self.dockarea.findChildren(LaserImageDock):
                for isotope in tool.calibrations.keys():
                    if isotope in dock.laser.isotopes:
                        dock.laser.data[isotope].calibration = copy.copy(
                            tool.calibrations[isotope]
                        )
                dock.draw()

        docks = self.dockarea.orderedDocks(self.dockarea.visibleDocks(LaserImageDock))
        cali_tool = StandardsTool(docks[0], self.viewoptions, parent=self)
        cali_tool.applyPressed.connect(applyTool)
        cali_tool.show()

    def menuOperationsTool(self) -> None:
        def applyTool(tool: Tool) -> None:
            if isinstance(tool.dock.laser, KrissKross):
                tool.dock.laser.data[tool.name] = KrissKrossData(tool.data)
            else:
                tool.dock.laser.data[tool.name] = LaserData(tool.data)
            tool.dock.populateComboIsotopes()
            tool.updateComboIsotopes()

        docks = self.dockarea.orderedDocks(self.dockarea.visibleDocks(LaserImageDock))
        op_tool = get_operations_tool(docks[0], self.viewoptions, parent=self)
        op_tool.applyPressed.connect(applyTool)
        op_tool.show()

    def menuColormap(self, action: QtWidgets.QAction) -> None:
        text = action.text().replace("&", "")
        self.viewoptions.colors.set_cmap(text)
        self.refresh()

    def menuColormapRange(self) -> None:
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
        if dlg.open():
            applyDialog(dlg)

    def menuInterpolation(self, action: QtWidgets.QAction) -> None:
        self.viewoptions.image.interpolation = action.text().replace("&", "")
        self.refresh()

    # def menuFiltering(self, action: QtWidgets.QAction) -> None:
    #     self.viewconfig["filtering"]["type"] = action.text().replace("&", "")
    #     self.refresh()

    # def menuFilteringProperties(self) -> None:
    #     def applyDialog(dialog: dialogs.ApplyDialog) -> None:
    #         self.viewconfig["filtering"]["window"] = dialog.window
    #         self.viewconfig["filtering"]["threshold"] = dialog.threshold
    #         self.refresh()

    #     dlg = dialogs.FilteringDialog(
    #         self.viewconfig["filtering"]["window"],
    #         self.viewconfig["filtering"]["threshold"],
    #         parent=self,
    #     )
    #     dlg.applyPressed.connect(applyDialog)
    #     if dlg.open():
    #         applyDialog(dlg)

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
        for dock in self.dockarea.findChildren(LaserImageDock):
            dock.canvas.options["colorbar"] = checked
            dock.canvas.redrawFigure()
            dock.draw()

    def menuOptionLabel(self, checked: bool) -> None:
        for dock in self.dockarea.findChildren(LaserImageDock):
            dock.canvas.options["label"] = checked
            dock.draw()

    def menuOptionScalebar(self, checked: bool) -> None:
        for dock in self.dockarea.findChildren(LaserImageDock):
            dock.canvas.options["scalebar"] = checked
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

    def closeEvent(self, event: QtCore.QEvent) -> None:
        for dock in self.dockarea.findChildren(LaserImageDock):
            dock.close()
        super().closeEvent(event)
