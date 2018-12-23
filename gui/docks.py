import numpy as np
import os.path

from PyQt5 import QtCore, QtGui, QtWidgets

from util.exporter import exportCsv, exportNpz, exportPng, exportVtr

from gui.canvas import Canvas
from gui.dialogs import CalibrationDialog, ConfigDialog, ExportDialog, TrimDialog

from typing import List, Union
from util.laser import LaserData
from util.krisskross import KrissKrossData
from gui.dialogs import ApplyDialog


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

        self.laser: LaserData = None
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
        self.action_export.setStatusTip("Export data to different formats.")
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
        isotope = self.combo_isotope.currentText()
        viewconfig = self.window().viewconfig
        self.canvas.plot(self.laser, isotope, viewconfig)

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
        if isinstance(self, KrissKrossImageDock):
            copy = KrissKrossImageDock(self.laser, self.parent())
        elif isinstance(self, LaserImageDock):
            copy = LaserImageDock(self.laser, self.parent())
        else:
            copy = ImageDock(self.parent())
        copy.draw()
        self.parent().smartSplitDock(self, copy)

    def onMenuSave(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", "", "Numpy archive(*.npz);;All files(*)"
        )
        if path:
            exportNpz(path, [self.laser])

    def onMenuExport(self) -> None:
        dlg = ExportDialog(
            [self.laser],
            default_path=os.path.dirname(self.laser.source) + "export.csv",
            default_isotope=self.combo_isotope.currentText(),
            parent=self,
        )
        if not dlg.exec():
            return

        if dlg.check_isotopes.isEnabled():
            isotopes: Union[List[str], List[None]] = (
                dlg.isotopes
                if dlg.check_isotopes.isChecked()
                else [dlg.combo_isotopes.currentText()]
            )
        else:
            isotopes = [None]

        if dlg.check_layers.isEnabled() and dlg.check_layers.isChecked():
            layers: Union[List[int], List[None]] = list(range(1, dlg.layers + 1))
        else:
            layers = [None]

        prompt_overwrite = True
        for isotope in isotopes:
            for layer in layers:
                path = dlg.getPath(isotope=isotope, layer=layer)
                result = self._export(
                    path,
                    isotope=isotope,
                    layer=layer,
                    prompt_overwrite=prompt_overwrite,
                )
                if result == QtWidgets.QMessageBox.No:
                    continue
                elif result == QtWidgets.QMessageBox.NoToAll:
                    return
                elif result == QtWidgets.QMessageBox.YesToAll:
                    prompt_overwrite = False

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
                # Only LaserImageDock, no KrissKrossImageDock
                docks = self.parent().findChildren(LaserImageDock)
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


class LaserImageDock(ImageDock):
    def __init__(self, laserdata: LaserData, parent: QtWidgets.QWidget = None):

        super().__init__(parent)
        self.laser = laserdata
        self.combo_isotope.addItems(self.laser.isotopes())
        self.setWindowTitle(self.laser.name)

    def _export(
        self,
        path: str,
        isotope: str = None,
        layer: int = None,
        prompt_overwrite: bool = True,
    ) -> QtWidgets.QMessageBox.StandardButton:
        if isotope is None:
            isotope = self.combo_isotope.currentText()

        result = QtWidgets.QMessageBox.Yes
        if prompt_overwrite and os.path.exists(path):
            result = QtWidgets.QMessageBox.warning(
                self,
                "Overwrite File?",
                f'The file "{os.path.basename(path)}" '
                "already exists. Do you wish to overwrite it?",
                QtWidgets.QMessageBox.Yes
                | QtWidgets.QMessageBox.YesToAll
                | QtWidgets.QMessageBox.No,
            )
            if result == QtWidgets.QMessageBox.No:
                return result

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            exportCsv(
                path,
                self.laser.get(isotope, calibrated=True, trimmed=True),
                isotope,
                self.laser.config,
            )
        elif ext == ".npz":
            exportNpz(path, [self.laser])
        elif ext == ".png":
            exportPng(
                path,
                self.laser.get(isotope, calibrated=True, trimmed=True),
                isotope,
                self.laser.aspect(),
                self.laser.extent(trimmed=True),
                self.window().viewconfig,
            )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Format",
                f"Unknown extention for '{os.path.basename(path)}'.",
            )
            return QtWidgets.QMessageBox.NoToAll

        return result


class KrissKrossImageDock(LaserImageDock):
    def __init__(self, kkdata: KrissKrossData, parent: QtWidgets.QWidget = None):

        super().__init__(kkdata, parent)
        self.setWindowTitle(f"kk:{self.laser.name}")

        # Config cannot be changed for krisskross images
        self.action_config.setEnabled(False)
        self.action_trim.setEnabled(False)

    def onMenuConfig(self) -> None:
        pass

    def onMenuTrim(self) -> None:
        pass

    def _export(
        self,
        path: str,
        isotope: str = None,
        layer: int = None,
        prompt_overwrite: bool = True,
    ) -> QtWidgets.QMessageBox.StandardButton:
        if isotope is None:
            isotope = self.combo_isotope.currentText()

        result = QtWidgets.QMessageBox.Yes
        if prompt_overwrite and os.path.exists(path):
            result = QtWidgets.QMessageBox.warning(
                self,
                "Overwrite File?",
                f'The file "{os.path.basename(path)}" '
                "already exists. Do you wish to overwrite it?",
                QtWidgets.QMessageBox.Yes
                | QtWidgets.QMessageBox.YesToAll
                | QtWidgets.QMessageBox.No,
            )
            if result == QtWidgets.QMessageBox.No:
                return result

        if layer is None:
            export_data = self.laser.get(isotope, calibrated=True, flattened=True)
        else:
            export_data = self.laser.get(isotope, calibrated=True, flattened=False)[
                :, :, layer - 1
            ]

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            np.savetxt(path, export_data, delimiter=",")
            exportCsv(path, export_data, isotope, self.laser.config)
        elif ext == ".npz":
            exportNpz(path, [self.laser])
        elif ext == ".png":
            exportPng(
                path,
                export_data,
                isotope,
                self.laser.aspect(),
                self.laser.extent(trimmed=True),
                self.window().viewconfig,
            )
        elif ext == ".vtr":
            exportVtr(path, self.laser)
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Format",
                f'Unknown extention for "{os.path.basename(path)}".',
            )
            return QtWidgets.QMessageBox.NoToAll

        return result
