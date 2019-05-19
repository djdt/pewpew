import os.path
import copy

from PyQt5 import QtCore, QtGui, QtWidgets

from pewpew.ui.widgets import Canvas, OverwriteFilePrompt
from pewpew.ui.dialogs import CalibrationDialog, ConfigDialog, StatsDialog
from pewpew.ui.dialogs.export import CSVExportDialog, PNGExportDialog

from laserlib import io
from laserlib.laser import Laser

from pewpew.ui.dialogs import ApplyDialog
from pewpew.ui.dialogs.export import ExportDialog


class ImageDockTitleBar(QtWidgets.QWidget):

    nameChanged = QtCore.pyqtSignal("QString")

    def __init__(self, title: str, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.title = QtWidgets.QLabel(title)
        # self.parent().windowTitleChanged.connect(self.setTitle)

        # Button Bar
        self.button_select_rect = QtWidgets.QPushButton(
            QtGui.QIcon.fromTheme("draw-rectangle"), ""
        )
        self.button_select_lasso = QtWidgets.QPushButton(
            QtGui.QIcon.fromTheme("edit-select-lasso"), ""
        )
        self.button_zoom = QtWidgets.QPushButton(QtGui.QIcon.fromTheme("zoom-in"), "")
        self.button_zoom.setToolTip("Zoom into slected area.")
        self.button_zoom_original = QtWidgets.QPushButton(
            QtGui.QIcon.fromTheme("zoom-original"), ""
        )
        self.button_zoom_original.setToolTip("Reset to original zoom.")
        self.button_close = QtWidgets.QPushButton(
            QtGui.QIcon.fromTheme("edit-delete"), ""
        )
        self.button_close.setToolTip("Close the image.")

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

        layout_buttons = QtWidgets.QHBoxLayout()
        layout_buttons.addWidget(self.button_select_rect, 0, QtCore.Qt.AlignLeft)
        layout_buttons.addWidget(self.button_select_lasso, 0, QtCore.Qt.AlignLeft)
        layout_buttons.addWidget(line)
        layout_buttons.addWidget(self.button_zoom, 0, QtCore.Qt.AlignRight)
        layout_buttons.addWidget(self.button_zoom_original, 0, QtCore.Qt.AlignRight)
        layout_buttons.addWidget(self.button_close, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.title, 1)
        layout.addLayout(layout_buttons, 0)

        self.setLayout(layout)

    def setTitle(self, title: str) -> None:
        self.button_select_rect = QtWidgets.QPushButton(
            QtGui.QIcon.fromTheme("draw-rectangle"), ""
        )
        if "&" not in title:
            self.title.setText(title)

    def mouseDoubleClickEvent(self, event: QtCore.QEvent) -> None:
        if self.title.underMouse():
            name, ok = QtWidgets.QInputDialog.getText(
                self, "Rename", "Name:", QtWidgets.QLineEdit.Normal, self.title.text()
            )
            if ok:
                self.nameChanged.emit(name)


class LaserImageDock(QtWidgets.QDockWidget):
    def __init__(self, laser: Laser, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
        )

        self.laser = laser
        self.canvas = Canvas(viewconfig=self.window().viewconfig, parent=self)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.onComboIsotope)
        self.combo_isotope.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.populateComboIsotopes()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.combo_isotope, 1, QtCore.Qt.AlignRight)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

        # Title bar
        self.title_bar = ImageDockTitleBar("", self)
        self.title_bar.nameChanged.connect(self.titleNameChanged)
        self.windowTitleChanged.connect(self.title_bar.setTitle)
        self.setTitleBarWidget(self.title_bar)
        self.setWindowTitle(self.laser.name)

        self.title_bar.button_select_lasso.clicked.connect(self.canvas.startLasso)
        self.title_bar.button_zoom.clicked.connect(self.canvas.startZoom)
        self.title_bar.button_zoom_original.clicked.connect(self.canvas.unzoom)
        self.title_bar.button_close.clicked.connect(self.onMenuClose)

        # Context menu actions
        self.action_copy_image = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy Image", self
        )
        self.action_copy_image.setStatusTip("Copy image to clipboard.")
        self.action_copy_image.triggered.connect(self.onMenuCopyImage)
        self.action_copy = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("edit-copy"), "Open Copy", self
        )
        self.action_copy.setStatusTip("Open a copy.")
        self.action_copy.triggered.connect(self.onMenuCopy)
        self.action_save = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("document-save"), "Save", self
        )
        self.action_save.setStatusTip("Save data to archive.")
        self.action_save.triggered.connect(self.onMenuSave)

        self.action_export = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("document-save-as"), "Export", self
        )
        self.action_export.setStatusTip("Save data to different formats.")
        self.action_export.triggered.connect(self.onMenuExport)

        self.action_calibration = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("go-top"), "Calibration", self
        )
        self.action_calibration.setStatusTip("Edit image calibration.")
        self.action_calibration.triggered.connect(self.onMenuCalibration)

        self.action_config = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("document-edit"), "Config", self
        )
        self.action_config.setStatusTip("Edit image config.")
        self.action_config.triggered.connect(self.onMenuConfig)

        self.action_stats = QtWidgets.QAction(
            QtGui.QIcon.fromTheme(""), "Statistics", self
        )
        self.action_stats.setStatusTip("Data histogram and statistics.")
        self.action_stats.triggered.connect(self.onMenuStats)
        self.action_stats.setEnabled(False)

        self.action_close = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("edit-delete"), "Close", self
        )
        self.action_close.setStatusTip("Close the images.")
        self.action_close.triggered.connect(self.onMenuClose)

    def draw(self) -> None:
        self.canvas.drawLaser(self.laser, self.combo_isotope.currentText())
        self.canvas.draw_idle()

    def buildContextMenu(self) -> QtWidgets.QMenu:
        context_menu = QtWidgets.QMenu(self)
        context_menu.addAction(self.action_copy_image)
        # context_menu.addAction(self.action_copy)
        context_menu.addSeparator()
        context_menu.addAction(self.action_save)
        context_menu.addAction(self.action_export)
        context_menu.addSeparator()
        context_menu.addAction(self.action_calibration)
        context_menu.addAction(self.action_config)
        context_menu.addAction(self.action_stats)
        context_menu.addSeparator()
        context_menu.addAction(self.action_close)
        return context_menu

    def populateComboIsotopes(self) -> None:
        isotopes = sorted(self.laser.isotopes())
        self.combo_isotope.blockSignals(True)
        self.combo_isotope.clear()
        self.combo_isotope.addItems(isotopes)
        self.combo_isotope.blockSignals(False)

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        context_menu = self.buildContextMenu()
        context_menu.exec(event.globalPos())

    def onMenuCopyImage(self) -> None:
        # The y axis is inverted so we must invert the bounds
        y = self.canvas.size().height()
        bbox = self.canvas.figure.get_tightbbox(self.canvas.get_renderer()).transformed(
            self.canvas.figure.dpi_scale_trans
        )
        x0, y0, w, h = bbox.bounds
        y0 = y - y0 - h
        QtWidgets.QApplication.clipboard().setPixmap(
            self.canvas.grab(QtCore.QRect(int(x0), int(y0), int(w), int(h)))
        )

    def onMenuCopy(self) -> None:
        laser_copy = copy.deepcopy(self.laser)
        dock_copy = type(self)(laser_copy, self.parent())
        self.parent().smartSplitDock(self, dock_copy)
        dock_copy.draw()

    def onMenuSave(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", "", "Numpy archive(*.npz);;All files(*)"
        )
        if path:
            io.npz.save(path, [self.laser])

    def onMenuExport(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export",
            "",
            "CSV files(*.csv);;Numpy archives(*.npz);;"
            "PNG images(*.png);;VTK Images(*.vti);;All files(*)",
            options=QtWidgets.QFileDialog.DontConfirmOverwrite,
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()

        if ext in [".npz", ".vti"]:  # Show an overwrite dialog
            if not OverwriteFilePrompt.promptOverwriteSingleFile(path, self):
                return self.onMenuExport()

        if ext == ".csv":
            dlg: ExportDialog = CSVExportDialog(
                path,
                name=self.combo_isotope.currentText(),
                names=len(self.laser.isotopes()),
                layers=1,
                parent=self,
            )
            if dlg.exec():
                paths = dlg.generate_paths(self.laser)
                kwargs = {"calibrate": self.window().viewconfig["calibrate"]}
                if dlg.options.trimmedChecked():
                    kwargs["extent"] = self.canvas.view
                for path, isotope, _ in paths:
                    if dlg.options.headerChecked():
                        header = io.csv.make_header(self.laser, isotope)
                    else:
                        header = ""
                    io.csv.save(path, self.laser.get(isotope, **kwargs), header=header)

        elif ext == ".npz":
            io.npz.save(path, [self.laser])
        elif ext == ".png":
            dlg = PNGExportDialog(
                path,
                name=self.combo_isotope.currentText(),
                names=len(self.laser.isotopes()),
                layers=1,
                parent=self,
            )
            if dlg.exec():
                paths = dlg.generate_paths(self.laser)
                old_size = self.canvas.figure.get_size_inches()
                size = dlg.options.imagesize()
                dpi = self.canvas.figure.get_dpi()
                self.canvas.figure.set_size_inches(size[0] / dpi, size[1] / dpi)

                for path, isotope, _ in paths:
                    self.canvas.drawLaser(self.laser, isotope)
                    self.canvas.figure.savefig(path, transparent=True, frameon=False)

                self.canvas.figure.set_size_inches(*old_size)
                self.canvas.drawLaser(self.laser, self.combo_isotope.currentText())
                self.canvas.draw()
        elif ext == ".vti":
            spacing = *self.laser.config.pixel_size(), self.laser.config.spotsize / 2.0
            io.vtk.save(
                path,
                self.laser.get_structured(
                    calibrate=self.window().viewconfig["calibrate"]
                ),
                spacing=spacing,
            )
        else:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Format", f"Unable to export {ext} format."
            )
            return self.onMenuExport()

    def onMenuCalibration(self) -> None:
        def applyDialog(dialog: ApplyDialog) -> None:
            if dialog.check_all.isChecked():
                docks = self.parent().findChildren(LaserImageDock)
            else:
                docks = [self]
            for dock in docks:
                for isotope in dlg.calibrations.keys():
                    if isotope in dock.laser.isotopes():
                        dock.laser.data[isotope].calibration = copy.copy(
                            dlg.calibrations[isotope]
                        )
                dock.draw()

        calibrations = {
            isotope: self.laser.data[isotope].calibration
            for isotope in self.laser.data.keys()
        }
        dlg = CalibrationDialog(
            calibrations, self.combo_isotope.currentText(), parent=self
        )
        dlg.applyPressed.connect(applyDialog)
        if dlg.exec():
            applyDialog(dlg)

    def onMenuConfig(self) -> None:
        def applyDialog(dialog: ApplyDialog) -> None:
            if dialog.check_all.isChecked():
                docks = self.parent().findChildren(LaserImageDock)
            else:
                docks = [self]
            for dock in docks:
                if type(dock.laser) == Laser:
                    dock.laser.config = copy.copy(dialog.config)
                    dock.draw()

        dlg = ConfigDialog(self.laser.config, parent=self)
        dlg.applyPressed.connect(applyDialog)
        if dlg.exec():
            applyDialog(dlg)

    def onMenuStats(self) -> None:
        return
        dlg = StatsDialog(self.laser, self.window().viewconfig, parent=self)
        dlg.exec()

    # def onMenuCalculate(self) -> None:
    #     def applyDialog(dialog: ApplyDialog) -> None:
    #         if dialog.data is None or hasattr(self.laser.data, dialog.data.name):
    #             return
    #         self.laser.data[dialog.data.name] = dialog.data
    #         self.populateComboIsotopes()

    #     dlg = CalculateDialog(self.laser, parent=self)
    #     dlg.applyPressed.connect(applyDialog)
    #     if dlg.exec():
    #         applyDialog(dlg)

    def onMenuClose(self) -> None:
        self.close()

    def onComboIsotope(self, text: str) -> None:
        self.draw()

    def titleNameChanged(self, name: str) -> None:
        self.setWindowTitle(name)
        if self.laser is not None:
            self.laser.name = name
