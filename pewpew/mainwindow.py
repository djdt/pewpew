import sys
import logging

from PySide2 import QtWidgets

from pewpew import __version__

from pewpew.actions import qAction, qActionGroup
from pewpew.log import LoggingDialog
from pewpew.widgets import dialogs
from pewpew.widgets.exportdialogs import ExportAllDialog
from pewpew.widgets.laser import LaserWidget, LaserViewSpace

from pewpew.widgets.tools import (
    ToolWidget,
    EditTool,
    StandardsTool,
    OverlayTool,
)
from pewpew.widgets.wizards import ImportWizard, SpotImportWizard, SRRImportWizard

from types import TracebackType

logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.resize(1280, 800)

        self.log = LoggingDialog()

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

    def createActions(self) -> None:
        self.action_about = qAction(
            "help-about", "&About", "About pew².", self.actionAbout
        )
        self.action_colormap_range = qAction(
            "", "Set &Range", "Set the range of the colormap.", self.actionColormapRange
        )
        self.action_colormap_range.setShortcut("Ctrl+R")
        self.action_config = qAction(
            "document-edit",
            "Default Config",
            "Edit the default config.",
            self.actionConfig,
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
            "insert-text", "Fontsize", "Set size of fonts.", self.actionFontsize
        )
        self.action_group_colormap = qActionGroup(
            self,
            list(self.viewspace.options.image.COLORMAPS.keys()),
            self.actionGroupColormap,
            checked=self.viewspace.options.image.get_cmap_name(),
            statuses=list(self.viewspace.options.image.COLORMAP_DESCRIPTIONS.values()),
        )
        self.action_group_interp = qActionGroup(
            self,
            list(self.viewspace.options.image.INTERPOLATIONS.keys()),
            self.actionGroupInterp,
            checked=self.viewspace.options.image.get_interpolation_name(),
        )
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
        self.action_wizard_spot.setEnabled(False)
        self.action_wizard_srr = qAction(
            "",
            "Kriss Kross Wizard",
            "Start the Super-Resolution-Reconstruction import wizard.",
            self.actionWizardSRR,
        )
        self.action_log = qAction(
            "clock", "&Show Log", "Show the pew² event and error log.", self.actionLog
        )
        self.action_open = qAction(
            "document-open", "&Open", "Open new document(s).", self.actionOpen
        )
        self.action_open.setShortcut("Ctrl+O")

        self.action_toggle_calibrate = qAction(
            "go-top", "Ca&librate", "Toggle calibration.", self.actionToggleCalibrate
        )
        self.action_toggle_calibrate.setShortcut("Ctrl+L")
        self.action_toggle_calibrate.setCheckable(True)
        self.action_toggle_calibrate.setChecked(self.viewspace.options.calibrate)
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
        self.action_tool_standards = qAction(
            "document-properties",
            "Standards Tool",
            "Open the standards calibration tool.",
            self.actionToolStandards,
        )
        # self.action_tool_calculations = qAction(
        #     "document-properties",
        #     "Calculations Tool",
        #     "Open the calculations tool.",
        #     self.actionToolCalculations,
        # )
        self.action_tool_edit = qAction(
            "document-properties",
            "Edit Tool",
            "Open tool for editing and transforming laser data.",
            self.actionToolEdit,
        )
        self.action_tool_overlay = qAction(
            "document-properties",
            "Image Overlay Tool",
            "Open the overlay tool.",
            self.actionToolOverlay,
        )

        self.action_crop_view = qAction(
            "transform-crop",
            "Crop to View",
            "Crop the image to the current view.",
            self.actionCropView,
        )
        self.action_crop_selection = qAction(
            "transform-crop",
            "Crop to Selection",
            "Crop the image to the current selection.",
            self.actionCropSelection,
        )

        self.action_transform_flip_horizontal = qAction(
            "object-flip-horizontal",
            "Flip Horizontal",
            "Flip the image about vertical axis.",
            self.actionTransformFlipHorz,
        )
        self.action_transform_flip_vertical = qAction(
            "object-flip-vertical",
            "Flip Vertical",
            "Flip the image about horizontal axis.",
            self.actionTransformFlipVert,
        )
        self.action_transform_rotate_left = qAction(
            "object-rotate-left",
            "Rotate Left",
            "Rotate the image 90° counter clockwise.",
            self.actionTransformRotateLeft,
        )
        self.action_transform_rotate_right = qAction(
            "object-rotate-right",
            "Rotate Right",
            "Rotate the image 90° clockwise.",
            self.actionTransformRotateRight,
        )

        self.action_refresh = qAction(
            "view-refresh", "Refresh", "Redraw documents.", self.refresh
        )
        self.action_refresh.setShortcut("F5")

    def actionAbout(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Information,
            "About pew²",
            (
                "Visualiser / converter for LA-ICP-MS data.\n"
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

    def actionColormapRange(self) -> QtWidgets.QDialog:
        def applyDialog(dialog: dialogs.ApplyDialog) -> None:
            self.viewspace.options.colors._ranges = dialog.ranges
            self.viewspace.options.colors.default_range = dialog.default_range
            self.refresh()

        dlg = dialogs.ColorRangeDialog(
            self.viewspace.options,
            self.viewspace.uniqueIsotopes(),
            current_isotope=self.viewspace.currentIsotope(),
            parent=self,
        )
        dlg.combo_isotope.currentTextChanged.connect(self.viewspace.setCurrentIsotope)
        dlg.applyPressed.connect(applyDialog)
        dlg.open()
        return dlg

    def actionConfig(self) -> QtWidgets.QDialog:
        dlg = dialogs.ConfigDialog(self.viewspace.config, parent=self)
        dlg.check_all.setChecked(True)
        dlg.check_all.setEnabled(False)
        dlg.configApplyAll.connect(self.viewspace.applyConfig)
        dlg.open()
        return dlg

    def actionExportAll(self) -> QtWidgets.QDialog:
        widgets = [
            w
            for v in self.viewspace.views
            for w in v.widgets()
            if isinstance(w, LaserWidget)
        ]
        dlg = ExportAllDialog(widgets, self)
        dlg.open()
        return dlg

    def actionFontsize(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QInputDialog(self)
        dlg.setWindowTitle("Fontsize")
        dlg.setLabelText("Fontisze:")
        dlg.setIntValue(self.viewspace.options.font.size)
        dlg.setIntRange(0, 100)
        dlg.setInputMode(QtWidgets.QInputDialog.IntInput)
        dlg.intValueSelected.connect(self.viewspace.options.font.set_size)
        dlg.intValueSelected.connect(self.refresh)
        dlg.open()
        return dlg

    def actionGroupColormap(self, action: QtWidgets.QAction) -> None:
        text = action.text().replace("&", "")
        self.viewspace.options.image.set_cmap_name(text)
        self.refresh()

    def actionCropView(self) -> None:
        widget = self.viewspace.activeWidget()
        if widget is None:
            return
        widget.crop()

    def actionCropSelection(self) -> None:
        widget = self.viewspace.activeWidget()
        if widget is None:
            return
        widget.cropToSelection()

    def actionGroupInterp(self, action: QtWidgets.QAction) -> None:
        text = action.text().replace("&", "")
        interp = self.viewspace.options.image.INTERPOLATIONS[text]
        self.viewspace.options.image.interpolation = interp
        self.refresh()

    def actionWizardImport(self) -> QtWidgets.QWizard:
        wiz = ImportWizard(config=self.viewspace.config, parent=self)
        wiz.laserImported.connect(self.viewspace.activeView().addLaser)
        wiz.open()
        return wiz

    def actionWizardSpot(self) -> QtWidgets.QWizard:
        wiz = SpotImportWizard(config=self.viewspace.config, parent=self)
        wiz.laserImported.connect(self.viewspace.activeView().addLaser)
        wiz.open()
        return wiz

    def actionWizardSRR(self) -> QtWidgets.QWizard:
        wiz = SRRImportWizard(config=self.viewspace.config, parent=self)
        wiz.laserImported.connect(self.viewspace.activeView().addLaser)
        wiz.open()
        return wiz

    def actionLog(self) -> None:
        self.log.show()

    def actionOpen(self) -> QtWidgets.QDialog:
        view = self.viewspace.activeView()
        return view.actionOpen()

    def actionToggleCalibrate(self, checked: bool) -> None:
        self.viewspace.options.calibrate = checked
        self.refresh()

    def actionToggleColorbar(self, checked: bool) -> None:
        self.viewspace.options.canvas.colorbar = checked
        # Hard refresh
        for view in self.viewspace.views:
            for widget in view.widgets():
                widget.canvas.drawFigure()
                widget.refresh()

    def actionToggleLabel(self, checked: bool) -> None:
        self.viewspace.options.canvas.label = checked
        self.refresh()

    def actionToggleScalebar(self, checked: bool) -> None:
        self.viewspace.options.canvas.scalebar = checked
        self.refresh()

    def actionToolStandards(self) -> None:
        widget = self.viewspace.activeWidget()
        index = widget.index
        if isinstance(widget, ToolWidget):
            widget = widget.widget
        tool = StandardsTool(widget)
        name = f"Standards: {widget.laser.name}"
        widget.view.removeTab(index)
        widget.view.insertTab(index, name, tool)
        tool.setActive()

    def actionToolEdit(self) -> None:
        widget = self.viewspace.activeWidget()
        index = widget.index
        if isinstance(widget, ToolWidget):
            widget = widget.widget
        tool = EditTool(widget)
        name = f"Edit: {widget.laser.name}"
        widget.view.removeTab(index)
        widget.view.insertTab(index, name, tool)
        tool.setActive()

    def actionToolOverlay(self) -> None:
        widget = self.viewspace.activeWidget()
        index = widget.index
        if isinstance(widget, ToolWidget):
            widget = widget.widget
        tool = OverlayTool(widget)
        name = f"Overlay: {widget.laser.name}"
        widget.view.removeTab(index)
        widget.view.insertTab(index, name, tool)
        tool.setActive()

    def actionTransformFlipHorz(self) -> None:
        widget = self.viewspace.activeWidget()
        if widget is None:
            return
        widget.transform(flip="horizontal")

    def actionTransformFlipVert(self) -> None:
        widget = self.viewspace.activeWidget()
        if widget is None:
            return
        widget.transform(flip="vertical")

    def actionTransformRotateLeft(self) -> None:
        widget = self.viewspace.activeWidget()
        if widget is None:
            return
        widget.transform(rotate="left")

    def actionTransformRotateRight(self) -> None:
        widget = self.viewspace.activeWidget()
        if widget is None:
            return
        widget.transform(rotate="right")

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

        menu_edit.addAction(self.action_crop_view)
        menu_edit.addAction(self.action_crop_selection)

        menu_edit.addSeparator()

        menu_edit.addAction(self.action_transform_flip_horizontal)
        menu_edit.addAction(self.action_transform_flip_vertical)
        menu_edit.addAction(self.action_transform_rotate_left)
        menu_edit.addAction(self.action_transform_rotate_right)

        menu_tools = self.menuBar().addMenu("&Tools")
        # menu_tools.addAction(self.action_tool_calculations)
        menu_tools.addAction(self.action_tool_edit)
        menu_tools.addAction(self.action_tool_standards)
        menu_tools.addAction(self.action_tool_overlay)

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
        menu_help.addAction(self.action_log)
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
        self.action_export_all.setEnabled(enabled)

        self.action_tool_edit.setEnabled(enabled)
        self.action_tool_standards.setEnabled(enabled)
        self.action_tool_overlay.setEnabled(enabled)

    def exceptHook(
        self, etype: type, value: BaseException, tb: TracebackType
    ) -> None:  # pragma: no cover
        if etype == KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting.")
            sys.exit(1)
        logger.exception("Uncaught exception", exc_info=(etype, value, tb))
        QtWidgets.QMessageBox.critical(self, "Uncaught Exception", str(value))
