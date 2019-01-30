import os.path
import copy

from PyQt5 import QtCore, QtGui, QtWidgets

from pewpew.ui.widgets import Canvas, OverwriteFilePrompt
from pewpew.ui.dialogs import CalibrationDialog, ConfigDialog, TrimDialog
from pewpew.ui.dialogs.export import CSVExportDialog, PNGExportDialog

from pewpew.lib import io

from pewpew.lib.laser import LaserData
from pewpew.ui.dialogs import ApplyDialog
from pewpew.ui.dialogs.export import ExportDialog


class ImageDockTitleBar(QtWidgets.QWidget):

    nameChanged = QtCore.pyqtSignal("QString")

    def __init__(self, title: str, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.title = QtWidgets.QLabel(title)
        self.parent().windowTitleChanged.connect(self.setTitle)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.title)

        self.setLayout(layout)

    def setTitle(self, title: str) -> None:
        if "&" not in title:
            self.title.setText(title)

    def mouseDoubleClickEvent(self, event: QtCore.QEvent) -> None:
        if self.title.underMouse():
            name, ok = QtWidgets.QInputDialog.getText(
                self, "Rename", "Name:", QtWidgets.QLineEdit.Normal, self.title.text()
            )
            if ok:
                self.nameChanged.emit(name)


class ImageDock(QtWidgets.QDockWidget):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
        )

        self.laser: LaserData = LaserData()
        self.canvas = Canvas(parent=self)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.onComboIsotope)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.combo_isotope, 1, QtCore.Qt.AlignRight)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

        self.title_bar = ImageDockTitleBar("test", self)
        self.title_bar.nameChanged.connect(self.titleNameChanged)
        self.setTitleBarWidget(self.title_bar)

        # Context menu actions
        self.action_copy = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("edit-copy"), "Open Copy", self
        )
        self.action_copy.setStatusTip("Open a copy of this data")
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
            QtGui.QIcon.fromTheme("document-properties"), "Config", self
        )
        self.action_config.setStatusTip("Edit image config.")
        self.action_config.triggered.connect(self.onMenuConfig)

        self.action_trim = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("edit-cut"), "Trim", self
        )
        self.action_trim.setStatusTip("Edit image trim.")
        self.action_trim.triggered.connect(self.onMenuTrim)

        self.action_close = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("edit-delete"), "Close", self
        )
        self.action_close.setStatusTip("Close the images.")
        self.action_close.triggered.connect(self.onMenuClose)

    def draw(self) -> None:
        self.canvas.clear()
        self.canvas.plot(
            self.laser, self.combo_isotope.currentText(), self.window().viewconfig
        )
        self.canvas.draw()

    def buildContextMenu(self) -> QtWidgets.QMenu:
        context_menu = QtWidgets.QMenu(self)
        context_menu.addAction(self.action_copy)
        context_menu.addSeparator()
        context_menu.addAction(self.action_save)
        context_menu.addAction(self.action_export)
        context_menu.addSeparator()
        context_menu.addAction(self.action_calibration)
        context_menu.addAction(self.action_config)
        context_menu.addAction(self.action_trim)
        context_menu.addSeparator()
        context_menu.addAction(self.action_close)
        return context_menu

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        context_menu = self.buildContextMenu()
        context_menu.exec(event.globalPos())

    def onMenuCopy(self) -> None:
        laser_copy = copy.deepcopy(self.laser)
        dock_copy = type(self)(laser_copy, self.parent())
        self.parent().smartSplitDock(self, dock_copy)

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
                isotope=self.combo_isotope.currentText(),
                isotopes=len(self.laser.isotopes()),
                layers=self.laser.layers(),
                parent=self,
            )
            if dlg.exec():
                paths, _ = dlg.generate_paths(self.laser)
                for path, isotope, _ in paths:
                    io.csv.save(
                        path,
                        self.laser,
                        isotope,
                        trimmed=dlg.options.hasTrimmed(),
                        include_header=dlg.options.hasHeader(),
                    )

        elif ext == ".npz":
            io.npz.save(path, [self.laser])
        elif ext == ".png":
            dlg = PNGExportDialog(
                path,
                isotope=self.combo_isotope.currentText(),
                isotopes=len(self.laser.isotopes()),
                layers=self.laser.layers(),
                parent=self,
            )
            if dlg.exec():
                paths, _ = dlg.generate_paths(self.laser)
                for path, isotope, _ in paths:
                    io.png.save(
                        path,
                        self.laser,
                        isotope,
                        self.window().viewconfig,
                        size=dlg.options.imagesize(),
                        include_colorbar=dlg.options.hasColorbar(),
                        include_scalebar=dlg.options.hasScalebar(),
                        include_label=dlg.options.hasLabel(),
                    )
        elif ext == ".vti":
            io.vtk.save(path, self.laser)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Format", f"Unable to export {ext} format."
            )
            return self.onMenuExport()

    def onMenuCalibration(self) -> None:
        def applyDialog(dialog: ApplyDialog) -> None:
            if dialog.check_all.isChecked():
                docks = self.parent().findChildren(ImageDock)
            else:
                docks = [self]
            for dock in docks:
                for isotope in dlg.calibration.keys():
                    if isotope in dock.laser.isotopes():
                        dock.laser.calibration[isotope] = dlg.calibration[isotope]
                dock.draw()

        dlg = CalibrationDialog(
            self.laser.calibration, self.combo_isotope.currentText(), parent=self
        )
        dlg.applyPressed.connect(applyDialog)
        if dlg.exec():
            applyDialog(dlg)

    def onMenuConfig(self) -> None:
        def applyDialog(dialog: ApplyDialog) -> None:
            if dialog.check_all.isChecked():
                # TODO see if this actually affects krisskross
                docks = self.parent().findChildren(ImageDock)
            else:
                docks = [self]
            for dock in docks:
                dock.laser.config["spotsize"] = dialog.spotsize
                dock.laser.config["speed"] = dialog.speed
                dock.laser.config["scantime"] = dialog.scantime
                dock.draw()

        dlg = ConfigDialog(self.laser.config, parent=self)
        dlg.applyPressed.connect(applyDialog)
        if dlg.exec():
            applyDialog(dlg)

    def onMenuTrim(self) -> None:
        def applyDialog(dialog: ApplyDialog) -> None:
            if dialog.check_all.isChecked():
                docks = self.parent().findChildren(ImageDock)
            else:
                docks = [self]
            for dock in docks:
                total = sum(
                    dock.laser.convertTrim(dialog.trim, dialog.combo_trim.currentText())
                )
                if total > dock.laser.data.shape[1]:
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Trim", "Trim larger than data."
                    )
                    return
            for dock in docks:
                dock.laser.setTrim(dialog.trim, dialog.combo_trim.currentText())
                dock.draw()

        dlg = TrimDialog(self.laser.trimAs("s"), parent=self)
        dlg.applyPressed.connect(applyDialog)

        if dlg.exec():
            applyDialog(dlg)

    def onMenuClose(self) -> None:
        self.close()

    def onComboIsotope(self, text: str) -> None:
        self.draw()

    def titleNameChanged(self, name: str) -> None:
        self.setWindowTitle(name)
        if self.laser is not None:
            self.laser.name = name
