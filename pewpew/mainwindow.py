import logging
import sys
from pathlib import Path
from types import TracebackType

from pewlib.config import Config, SpotConfig
from pewlib.io.laser import is_nwi_laser_log
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.actions import qAction, qActionGroup
from pewpew.help import HelpDialog
from pewpew.log import LoggingDialog
from pewpew.widgets import dialogs
from pewpew.widgets.exportdialogs import ExportAllDialog
from pewpew.widgets.laser import LaserTabView
from pewpew.widgets.wizards import ImportWizard, LaserLogImportWizard, SpotImportWizard
from pewpew.graphics.colortable import get_icon

logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    max_recent_files = 10
    """Pewpew mainwindow, holding a Lasertabview.
    Actions for the menu and status bars are created and stored here.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.resize(1280, 800)

        self.setAcceptDrops(True)

        self.log = LoggingDialog()
        self.help = HelpDialog()

        self.tabview = LaserTabView()
        self.tabview.fileImported.connect(self.updateRecentFiles)
        self.tabview.numTabsChanged.connect(self.updateActionAvailablity)
        self.tabview.numLaserItemsChanged.connect(self.updateActionAvailablity)
        self.setCentralWidget(self.tabview)

        self.createActions()
        self.createMenus()
        self.statusBar().showMessage(
            f"Welcome to pew² version {QtWidgets.QApplication.applicationVersion()}."
        )
        self.button_status_um = QtWidgets.QRadioButton("μ")
        self.button_status_index = QtWidgets.QRadioButton("i")
        self.button_status_um.setChecked(True)
        self.button_status_um.toggled.connect(self.buttonStatusUnit)
        self.button_status_index.toggled.connect(self.buttonStatusUnit)
        self.statusBar().addPermanentWidget(self.button_status_um)
        self.statusBar().addPermanentWidget(self.button_status_index)

        self.updateActionAvailablity()
        self.updateRecentFiles()

        self.default_config = Config()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
            if any(is_nwi_laser_log(path) for path in paths):
                event.acceptProposedAction()
        super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if not event.mimeData().hasUrls():
            return super().dropEvent(event)
        log_paths = []
        paths = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if is_nwi_laser_log(path):
                log_paths.append(path)
            else:
                paths.append(path)

        wiz = LaserLogImportWizard(
            path=log_paths[0],
            laser_paths=paths,
            options=self.tabview.options,
            parent=self,
        )
        wiz.laserImported.connect(self.tabview.importFile)
        wiz.laserImported.connect(
            lambda: self.tabview.activeWidget().graphics.zoomReset()
        )
        wiz.open()
        event.acceptProposedAction()

    def createActions(self) -> None:
        self.action_about = qAction(
            "help-about", "&About", "About pew².", self.actionAbout
        )

        self.action_colortable_range = qAction(
            "format-number-percent",
            "Set &Range",
            "Set the numerical range of the colortable.",
            self.dialogColortableRange,
        )
        self.action_colortable_range.setShortcut("Ctrl+R")

        self.action_config = qAction(
            "document-edit",
            "Default Config",
            "Edit the default config.",
            self.dialogConfig,
        )
        self.action_config.setShortcut("Ctrl+K")

        self.action_exit = qAction(
            "application-exit", "Quit", "Exit the program.", self.close
        )
        self.action_exit.setShortcut("Ctrl+Shift+Q")

        self.action_export_all = qAction(
            "document-save-all",
            "E&xport All",
            "Export all open documents.",
            self.actionExportAll,
        )
        self.action_export_all.setShortcut("Ctrl+Shift+X")

        self.action_fontsize = qAction(
            "insert-text",
            "Fontsize",
            "Set the font size in points.",
            self.dialogFontsize,
        )

        self.action_group_colortable = qActionGroup(
            self,
            list(self.tabview.options.colortables.keys()),
            self.actionGroupColortable,
            checked=self.tabview.options.colortable,
            statuses=list(self.tabview.options.colortables.values()),
            icons=[get_icon(c) for c in self.tabview.options.colortables.keys()],
        )

        self.action_help = qAction(
            "help-contents", "&Help", "Show the help contents.", self.help.open
        )

        self.action_log = qAction(
            "clock", "&Show Log", "Show the pew² event and error log.", self.log.open
        )

        self.action_nan_color = qAction(
            "color-picker",
            "Set NaN Color",
            "Set the color of not-a-number values in images.",
            self.actionDialogNaNColor,
        )

        self.action_open = qAction(
            "document-open", "&Open", "Open new document(s).", self.actionOpen
        )
        self.action_open.setShortcut("Ctrl+O")

        self.action_open_recent = QtGui.QActionGroup(self)
        self.action_open_recent.triggered.connect(
            lambda a: self.tabview.openDocument(a.text())
        )

        self.action_process = qAction(
            "view-process-tree",
            "Process Pipeline",
            "Apply processing to multiple images.",
            self.actionDialogProcess,
        )

        self.action_add_tab = qAction(
            "tab-new",
            "New Tab",
            "Create a workspace in a new tab.",
            self.tabview.newLaserTab,
        )
        self.action_add_tab.setShortcut("Ctrl+T")

        self.action_refresh = qAction(
            "view-refresh", "Refresh", "Redraw documents.", self.tabview.refresh
        )
        self.action_refresh.setShortcut("F5")

        self.action_smooth = qAction(
            "smooth",
            "&Smooth",
            "Smooth images with bilinear interpolation.",
            self.tabview.options.setSmoothing,
        )
        self.action_smooth.setCheckable(True)
        self.action_smooth.setChecked(self.tabview.options.smoothing)

        self.action_toggle_calibrate = qAction(
            "go-top",
            "Ca&librate",
            "Toggle calibration.",
            lambda checked: [
                setattr(self.tabview.options, "calibrate", checked),
                self.tabview.refresh,
            ],
        )
        self.action_toggle_calibrate.setShortcut("Ctrl+L")
        self.action_toggle_calibrate.setCheckable(True)
        self.action_toggle_calibrate.setChecked(self.tabview.options.calibrate)

        self.action_toggle_scalebar = qAction(
            "",
            "Show Scalebar",
            "Toggle the visibility of the scalebar.",
            self.tabview.options.setScalebarVisible,
        )
        self.action_toggle_scalebar.setCheckable(True)
        self.action_toggle_scalebar.setChecked(self.tabview.options.scalebar)

        self.action_wizard_import = qAction(
            "",
            "Import Wizard",
            "Start the line-wise import wizard.",
            self.actionWizardImport,
        )
        self.action_wizard_laserlog = qAction(
            "",
            "ESL Laser Log Wizard",
            "Import data and sync to a ESL ActiveView2 log file.",
            self.actionWizardLaserLog,
        )
        self.action_wizard_spot = qAction(
            "",
            "Spotwise Wizard",
            "Start the import wizard for data collected spot-wise.",
            self.actionWizardSpot,
        )
        # self.action_wizard_srr = qAction(
        #     "",
        #     "Kriss Kross Wizard",
        #     "Start the Super-Resolution-Reconstruction import wizard.",
        #     self.actionWizardSRR,
        # )

    def createMenus(self) -> None:
        # File
        menu_file = self.menuBar().addMenu("&File")
        menu_file.addAction(self.action_open)

        self.menu_recent = menu_file.addMenu("Open Recent")
        self.menu_recent.setIcon(QtGui.QIcon.fromTheme("document-open-recent"))
        self.menu_recent.setEnabled(False)

        # File -> Import
        menu_import = menu_file.addMenu("&Import")
        menu_import.addAction(self.action_wizard_import)
        menu_import.addAction(self.action_wizard_laserlog)
        menu_import.addAction(self.action_wizard_spot)
        # menu_import.addAction(self.action_wizard_srr)

        menu_file.addSeparator()

        menu_file.addAction(self.action_export_all)

        menu_file.addSeparator()

        menu_file.addAction(self.action_exit)

        # Edit
        menu_edit = self.menuBar().addMenu("&Edit")
        menu_edit.addAction(self.action_config)
        menu_edit.addAction(self.action_toggle_calibrate)

        menu_edit.addSeparator()

        menu_edit.addAction(self.action_process)

        menu_edit.addSeparator()

        menu_edit.addAction(self.action_add_tab)

        # View
        menu_view = self.menuBar().addMenu("&View")
        menu_cmap = menu_view.addMenu("&Colortable")
        menu_cmap.setIcon(QtGui.QIcon.fromTheme("color-management"))
        menu_cmap.setStatusTip("Colortable of displayed images.")
        menu_cmap.addActions(self.action_group_colortable.actions())
        menu_cmap.addAction(self.action_colortable_range)

        menu_view.addSeparator()

        menu_view.addAction(self.action_nan_color)

        # View - interpolation
        menu_view.addAction(self.action_smooth)

        menu_view.addAction(self.action_fontsize)

        menu_view.addSeparator()

        menu_view.addAction(self.action_toggle_scalebar)

        menu_view.addSeparator()

        menu_view.addAction(self.action_refresh)

        # Help
        menu_help = self.menuBar().addMenu("&Help")
        menu_help.addAction(self.action_log)
        menu_help.addAction(self.action_help)
        menu_help.addAction(self.action_about)

    def updateRecentFiles(self, new_path: Path | None = None) -> None:
        settings = QtCore.QSettings()
        num = settings.beginReadArray("RecentFiles")
        paths = []
        for i in range(num):
            settings.setArrayIndex(i)
            path = Path(settings.value("Path"))
            if path != new_path:
                paths.append(path)
        settings.endArray()

        if new_path is not None:
            paths.insert(0, new_path)
            paths = paths[: MainWindow.max_recent_files]

            settings.remove("RecentFiles")
            settings.beginWriteArray("RecentFiles", len(paths))
            for i, path in enumerate(paths):
                settings.setArrayIndex(i)
                settings.setValue("Path", str(path))
            settings.endArray()

        # Clear old actions
        self.menu_recent.clear()
        for action in self.action_open_recent.actions():
            self.action_open_recent.removeAction(action)

        # Add new
        self.menu_recent.setEnabled(len(paths) > 0)
        for path in paths:
            action = QtGui.QAction(str(path), self)
            self.action_open_recent.addAction(action)
            self.menu_recent.addAction(action)

    # === Actions ===
    def actionDialogColorTableRange(self) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.ColocalisationDialog` and apply result."""

        def applyDialog(dialog: dialogs.ApplyDialog) -> None:
            self.tabview.options._colorranges = dialog.ranges
            self.tabview.options.colorrange_default = dialog.default_range
            self.tabview.refresh()

        dlg = dialogs.ColorRangeDialog(
            self.tabview.options._colorranges,
            self.tabview.options.colorrange_default,
            self.uniqueElements(),
            current_element=self.currentElement(),
            parent=self,
        )
        dlg.combo_element.currentTextChanged.connect(self.setCurrentElement)
        dlg.applyPressed.connect(applyDialog)
        dlg.open()
        return dlg

    def actionDialogNaNColor(self) -> QtWidgets.QDialog:
        def applyDialog(color: QtGui.QColor) -> None:
            if color != self.tabview.options.nan_color:
                self.tabview.options.nan_color = color
                self.tabview.refresh()

        dlg = QtWidgets.QColorDialog(self.tabview.options.nan_color, parent=self)
        dlg.colorSelected.connect(applyDialog)  # type: ignore
        dlg.open()
        return dlg

    def actionDialogProcess(self) -> dialogs.ProcessingDialog:
        dlg = dialogs.ProcessingDialog(
            self.tabview.uniqueElements(), self.tabview.laserItems(), parent=self
        )
        dlg.open()
        return dlg

    def actionAbout(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Information,
            "About pew²",
            (
                "Import, process and export of LA-ICP-MS data.\n"
                f"Version {QtWidgets.QApplication.applicationVersion()}\n"
                "Developed by the Hyphenated Mass Spectrometry Laboratory.\n"
                "https://github.com/djdt/pewpew"
            ),
            parent=self,
        )
        if self.windowIcon() is not None:
            dlg.setIconPixmap(self.windowIcon().pixmap(64, 64))
        dlg.open()
        return dlg

    def actionExportAll(self) -> QtWidgets.QDialog:
        dlg = ExportAllDialog(self.tabview.laserItems(), self)
        dlg.open()
        return dlg

    def actionGroupColortable(self, action: QtGui.QAction) -> None:
        text = action.text().replace("&", "")
        self.tabview.options.colortable = text
        self.tabview.refresh()

    # def actionOpen(self) -> QtWidgets.QDialog:
    #     view = self.tabview.activeView()
    #     return view.actionOpen()

    def actionOpen(self) -> QtWidgets.QDialog:
        """Opens a file dialog for loading new lasers."""

        dir = QtCore.QSettings().value("RecentFiles/1/Path", None)
        dir = str(Path(dir).parent) if dir is not None else ""
        dlg = QtWidgets.QFileDialog(
            self,
            "Open File(s).",
            dir,
            "CSV Documents(*.csv *.txt *.text);;"
            "Images(*.bmp *.jpg *.jpeg *.png);;"
            "Numpy Archives(*.npz);;"
            "All files(*)",
        )
        dlg.selectNameFilter("All files(*)")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.filesSelected.connect(self.tabview.openDocument)
        dlg.open()
        return dlg

    def actionWizardImport(self) -> QtWidgets.QWizard:
        wiz = ImportWizard(config=self.tabview.config, parent=self)
        wiz.laserImported.connect(self.tabview.importFile)
        wiz.open()
        return wiz

    def actionWizardSpot(self) -> QtWidgets.QWizard:
        config = SpotConfig(self.tabview.config.spotsize, self.tabview.config.spotsize)
        wiz = SpotImportWizard(config=config, options=self.tabview.options, parent=self)
        wiz.laserImported.connect(self.tabview.importFile)
        wiz.open()
        return wiz

    def actionWizardLaserLog(self) -> QtWidgets.QWizard:
        wiz = LaserLogImportWizard(options=self.tabview.options, parent=self)
        wiz.laserImported.connect(self.tabview.importFile)
        wiz.laserImported.connect(
            lambda: self.tabview.activeWidget().graphics.zoomReset()
        )
        wiz.open()
        return wiz

    # def actionWizardSRR(self) -> QtWidgets.QWizard:
    #     wiz = SRRImportWizard(config=self.tabview.config, parent=self)
    #     wiz.laserImported.connect(self.tabview.importFile)
    #     wiz.open()
    #     return wiz

    def dialogColortableRange(self) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.ColorRangeDialog` and apply result."""

        def applyDialog(dialog: dialogs.ColorRangeDialog) -> None:
            self.tabview.options.color_ranges = dialog.ranges
            self.tabview.options.color_range_default = dialog.default_range
            self.tabview.refresh()

        item = self.tabview.focusLaserItem()
        dlg = dialogs.ColorRangeDialog(
            self.tabview.options.color_ranges,
            self.tabview.options.color_range_default,
            self.tabview.uniqueElements(),
            current_element=item.element() if item is not None else None,
            parent=self,
        )
        dlg.combo_element.currentTextChanged.connect(self.tabview.setElement)
        dlg.applyPressed.connect(applyDialog)
        dlg.open()
        return dlg

    def dialogConfig(self) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.ConfigDialog` and apply result."""
        # Todo Potential config item
        dlg = dialogs.ConfigDialog(self.default_config, parent=self)
        dlg.check_all.setChecked(True)
        dlg.check_all.setEnabled(False)
        dlg.configApplyAll.connect(self.tabview.applyConfig)
        dlg.open()
        return dlg

    def dialogFontsize(self) -> QtWidgets.QDialog:
        """Simple dialog for editing image font size."""
        dlg = QtWidgets.QInputDialog(self)
        dlg.setWindowTitle("Fontsize")
        dlg.setLabelText("Fontisze:")
        dlg.setIntValue(self.tabview.options.font.pointSize())
        dlg.setIntRange(2, 96)
        dlg.setInputMode(QtWidgets.QInputDialog.IntInput)
        dlg.intValueSelected.connect(self.tabview.options.setFontSize)
        dlg.intValueSelected.connect(self.tabview.refresh)
        dlg.open()
        return dlg

    def buttonStatusUnit(self, toggled: bool) -> None:
        """Callback for 'button_status_um'."""
        if self.button_status_um.isChecked():
            self.tabview.options.units = "μm"
        elif self.button_status_index.isChecked():
            self.tabview.options.units = "index"

    def updateActionAvailablity(self) -> None:
        """Enables tools if at least one view is present."""
        items = self.tabview.laserItems()
        self.action_export_all.setEnabled(len(items) > 0)

        # Tools require an active view
        # focus_item = self.tabview.focusLaserItem()
        # for action in self.actions_tools:
        #     action.setEnabled(len(items) > 0 and focus_item is not None)

    def exceptHook(
        self, etype: type, value: BaseException, tb: TracebackType
    ) -> None:  # pragma: no cover
        """Redirect errors to the log."""
        if etype == KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting.")
            sys.exit(1)
        logger.exception("Uncaught exception", exc_info=(etype, value, tb))
        QtWidgets.QMessageBox.critical(self, "Uncaught Exception", str(value))
