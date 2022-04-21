import sys
import logging
from typing import Optional

from PySide2 import QtGui, QtWidgets

from pewlib.config import SpotConfig

from pewpew import __version__

from pewpew.actions import qAction, qActionGroup
from pewpew.graphics.options import GraphicsOptions
from pewpew.log import LoggingDialog
from pewpew.help import HelpDialog
from pewpew.widgets import dialogs
from pewpew.widgets.exportdialogs import ExportAllDialog
from pewpew.widgets.laser import LaserTabWidget, LaserTabView

from pewpew.widgets.tools import (
    ToolWidget,
    CalculatorTool,
    DriftTool,
    FilteringTool,
    MergeTool,
    StandardsTool,
    OverlayTool,
)
from pewpew.widgets.wizards import ImportWizard, SpotImportWizard, SRRImportWizard

from types import TracebackType

logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    """Pewpew mainwindow, holding a Lasertabview.
    Actions for the menu and status bars are created and stored here.
    """

    ENABLED_TOOLS = {
        "Calculator": (CalculatorTool, None, ""),
        "Drift": (DriftTool, None, "Correct instrument drift."),
        "Filter": (
            FilteringTool,
            None,
            "Remove spikes and instrument noise from data.",
        ),
        # "Merge": (MergeTool, "align-vertical-top", "Tool for merging multiple images."),
        "Calibration": (StandardsTool, None, "Generate calibrations from stanards."),
        "Overlay": (OverlayTool, None, "Overlay elements as RGB images."),
    }

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.resize(1280, 800)

        self.log = LoggingDialog()
        self.help = HelpDialog()

        self.tabview = LaserTabView()
        self.tabview.numTabsChanged.connect(self.updateActionAvailablity)
        self.setCentralWidget(self.tabview)

        self.createActions()
        self.createMenus()
        self.statusBar().showMessage(f"Welcome to pew² version {__version__}.")
        self.button_status_um = QtWidgets.QRadioButton("μ")
        self.button_status_index = QtWidgets.QRadioButton("i")
        self.button_status_um.setChecked(True)
        self.button_status_um.toggled.connect(self.buttonStatusUnit)
        self.button_status_index.toggled.connect(self.buttonStatusUnit)
        self.statusBar().addPermanentWidget(self.button_status_um)
        self.statusBar().addPermanentWidget(self.button_status_index)

        self.updateActionAvailablity()

    def createActions(self) -> None:
        self.action_about = qAction(
            "help-about", "&About", "About pew².", self.actionAbout
        )
        self.action_help = qAction(
            "help-contents", "&Help", "Show the help contents.", self.help.open
        )
        self.action_colortable_range = qAction(
            "",
            "Set &Range",
            "Set the range of the colortable.",
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
        )
        self.action_smooth = qAction(
            "smooth",
            "&Smooth",
            "Smooth images with bilinear interpolation.",
            lambda checked: [
                setattr(self.tabview.options, "smoothing", checked),
                self.tabview.refresh,
            ],
        )
        self.action_smooth.setCheckable(True)
        self.action_smooth.setChecked(self.tabview.options.smoothing)
        self.action_wizard_import = qAction(
            "",
            "Import Wizard",
            "Start the line-wise import wizard. .",
            self.actionWizardImport,
        )
        self.action_wizard_spot = qAction(
            "",
            "Spotwise Wizard",
            "Start the import wizard for data collected spot-wise.",
            self.actionWizardSpot,
        )
        self.action_wizard_srr = qAction(
            "",
            "Kriss Kross Wizard",
            "Start the Super-Resolution-Reconstruction import wizard.",
            self.actionWizardSRR,
        )
        self.action_log = qAction(
            "clock", "&Show Log", "Show the pew² event and error log.", self.log.open
        )
        self.action_open = qAction(
            "document-open", "&Open", "Open new document(s).", self.actionOpen
        )
        self.action_open.setShortcut("Ctrl+O")

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

        self.action_overlay_items = [
            # qAction("", "Show Labels", "Toggle visiblity of labels.", lambda checked: setattr(self.tabview.options.overlay_items, "label", checked)),
            qAction(
                "",
                f"Show {item.capitalize()}",
                f"Toggle visibility of the {item}.",
                lambda checked: [
                    setattr(self.tabview.options.overlay_items, item, checked),
                    self.tabview.refresh,
                ],
            )
            for item in self.tabview.options.overlay_items.keys()
        ]
        for action, checked in zip(
            self.action_overlay_items, self.tabview.options.overlay_items.values()
        ):
            action.setCheckable(True)
            action.setChecked(checked)

        self.action_tools = [
            qAction(
                icon or "document-properties",
                name,
                desc,
                lambda: self.openTool(tool, name),
            )
            for name, (tool, icon, desc) in self.ENABLED_TOOLS.items()
        ]

        self.action_transforms = [
            qAction(
                "object-flip-horizontal",
                "Flip Horizontal",
                "Flip the image about vertical axis.",
                lambda: self.actionTransform(flip="horizontal"),
            ),
            qAction(
                "object-flip-vertical",
                "Flip Vertical",
                "Flip the image about horizontal axis.",
                lambda: self.actionTransform(flip="vertical"),
            ),
            qAction(
                "object-rotate-left",
                "Rotate Left",
                "Rotate the image 90° counter clockwise.",
                lambda: self.actionTransform(rotate="left"),
            ),
            qAction(
                "object-rotate-right",
                "Rotate Right",
                "Rotate the image 90° clockwise.",
                lambda: self.actionTransform(rotate="right"),
            ),
        ]

        self.action_refresh = qAction(
            "view-refresh", "Refresh", "Redraw documents.", self.tabview.refresh
        )
        self.action_refresh.setShortcut("F5")

    def createMenus(self) -> None:
        # File
        menu_file = self.menuBar().addMenu("&File")
        menu_file.addAction(self.action_open)
        # File -> Import
        menu_import = menu_file.addMenu("&Import")
        menu_import.addAction(self.action_wizard_import)
        menu_import.addAction(self.action_wizard_spot)
        menu_import.addAction(self.action_wizard_srr)

        menu_file.addSeparator()

        menu_file.addAction(self.action_export_all)

        menu_file.addSeparator()

        menu_file.addAction(self.action_exit)

        # Edit
        menu_edit = self.menuBar().addMenu("&Edit")
        menu_edit.addAction(self.action_config)
        menu_edit.addAction(self.action_toggle_calibrate)

        menu_edit.addSeparator()

        for action in self.action_transforms:
            menu_edit.addAction(action)

        # Tools
        menu_tools = self.menuBar().addMenu("&Tools")
        for action in self.action_tools:
            menu_tools.addAction(action)

        # View
        menu_view = self.menuBar().addMenu("&View")
        menu_cmap = menu_view.addMenu("&Colortable")
        menu_cmap.setIcon(QtGui.QIcon.fromTheme("color-management"))
        menu_cmap.setStatusTip("Colortable of displayed images.")
        menu_cmap.addActions(self.action_group_colortable.actions())
        menu_cmap.addAction(self.action_colortable_range)

        # View - interpolation
        menu_view.addAction(self.action_smooth)

        menu_view.addAction(self.action_fontsize)

        menu_view.addSeparator()

        for action in self.action_overlay_items:
            menu_view.addAction(action)

        menu_view.addSeparator()

        menu_view.addAction(self.action_refresh)

        # Help
        menu_help = self.menuBar().addMenu("&Help")
        menu_help.addAction(self.action_log)
        menu_help.addAction(self.action_help)
        menu_help.addAction(self.action_about)

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

    def actionAbout(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Information,
            "About pew²",
            (
                "Import, process and export of LA-ICP-MS data.\n"
                f"Version {__version__}\n"
                "Developed by the Atomic Medicine Initiative.\n"
                "https://github.com/djdt/pewpew"
            ),
            parent=self,
        )
        if self.windowIcon() is not None:
            dlg.setIconPixmap(self.windowIcon().pixmap(64, 64))
        dlg.open()
        return dlg

    def actionExportAll(self) -> QtWidgets.QDialog:
        widgets = [
            w
            for v in self.tabview.views
            for w in v.widgets()
            if isinstance(w, LaserTabWidget)
        ]
        dlg = ExportAllDialog(widgets, self)
        dlg.open()
        return dlg

    def actionGroupColortable(self, action: QtWidgets.QAction) -> None:
        text = action.text().replace("&", "")
        self.tabview.options.colortable = text
        self.tabview.refresh()

    def actionOpen(self) -> QtWidgets.QDialog:
        view = self.tabview.activeView()
        return view.actionOpen()

    def openTool(self, tool: ToolWidget, name: str) -> None:
        widget = self.tabview.activeWidget()
        if widget is None:
            return
        index = widget.index
        if isinstance(widget, ToolWidget):
            widget = widget.widget
        tool = tool(widget)
        name = f"{name}: {widget.laserName()}"
        widget.view.removeTab(index)
        widget.view.insertTab(index, name, tool)
        tool.activate()

    def actionTransform(
        self, flip: Optional[str] = None, rotate: Optional[str] = None
    ) -> None:
        widget = self.tabview.activeWidget()
        if widget is None:
            return
        widget.transform(flip=flip, rotate=rotate)

    def actionWizardImport(self) -> QtWidgets.QWizard:
        wiz = ImportWizard(config=self.tabview.config, parent=self)
        wiz.laserImported.connect(self.tabview.activeView().addLaser)
        wiz.open()
        return wiz

    def actionWizardSpot(self) -> QtWidgets.QWizard:
        config = SpotConfig(self.tabview.config.spotsize, self.tabview.config.spotsize)
        wiz = SpotImportWizard(config=config, options=self.tabview.options, parent=self)
        wiz.laserImported.connect(self.tabview.activeView().addLaser)
        wiz.open()
        return wiz

    def actionWizardSRR(self) -> QtWidgets.QWizard:
        wiz = SRRImportWizard(config=self.tabview.config, parent=self)
        wiz.laserImported.connect(self.tabview.activeView().addLaser)
        wiz.open()
        return wiz

    def dialogColortableRange(self) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.ColorRangeDialog` and apply result."""

        def applyDialog(dialog: dialogs.ApplyDialog) -> None:
            self.tabview.options.color_ranges = dialog.ranges
            self.tabview.options.color_range_default = dialog.default_range
            self.tabview.refresh()

        dlg = dialogs.ColorRangeDialog(
            self.tabview.options.color_ranges,
            self.tabview.options.color_range_default,
            self.uniqueElements(),
            current_element=self.currentElement(),
            parent=self,
        )
        dlg.combo_element.currentTextChanged.connect(self.setCurrentElement)
        dlg.applyPressed.connect(applyDialog)
        dlg.open()
        return dlg

    def dialogConfig(self) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.ConfigDialog` and apply result."""
        dlg = dialogs.ConfigDialog(self.config, parent=self)
        dlg.check_all.setChecked(True)
        dlg.check_all.setEnabled(False)
        dlg.configApplyAll.connect(self.applyConfig)
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
        dlg.intValueSelected.connect(self.tabview.options.font.setPointSize)
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
        enabled = self.tabview.tabs.count() > 0
        self.action_export_all.setEnabled(enabled)

        # Tools require an active view
        enabled = enabled and self.tabview.tabs.count() > 0

        for action in self.action_tools:
            action.setEnabled(enabled)

    def exceptHook(
        self, etype: type, value: BaseException, tb: TracebackType
    ) -> None:  # pragma: no cover
        """Redirect errors to the log."""
        if etype == KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting.")
            sys.exit(1)
        logger.exception("Uncaught exception", exc_info=(etype, value, tb))
        QtWidgets.QMessageBox.critical(self, "Uncaught Exception", str(value))
