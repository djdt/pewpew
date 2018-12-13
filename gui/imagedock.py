from PyQt5 import QtCore, QtGui, QtWidgets

from util.exporter import exportCsv, exportNpz, exportPng, exportVtr

from gui.canvas import Canvas
from gui.dialogs import CalibrationDialog, ConfigDialog, ExportDialog, TrimDialog

import numpy as np
import os.path


class ImageDockTitleBar(QtWidgets.QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)

        label = QtWidgets.QLineEdit(title)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)

        self.setLayout(layout)

    def mouseDoubleClickEvent(self, event):
        print("ASDASD")


class ImageDock(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
        )

        self.laser = None
        self.canvas = Canvas(parent=self)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.onComboIsotope)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.combo_isotope, 1, QtCore.Qt.AlignRight)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

        self.setTitleBarWidget(ImageDockTitleBar("test", self))

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

    def draw(self):
        self.canvas.clear()
        isotope = self.combo_isotope.currentText()
        viewconfig = self.window().viewconfig
        self.canvas.plot(self.laser, isotope, viewconfig)

    def buildContextMenu(self):
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

    def contextMenuEvent(self, event):
        context_menu = self.buildContextMenu()
        context_menu.exec(event.globalPos())

    def onMenuCopy(self):
        dock_copy = type(self)(self.laser, self.parent())
        dock_copy.draw()
        self.parent().smartSplitDock(self, dock_copy)

    def onMenuSave(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save", "", "Numpy archive(*.npz);;All files(*)"
        )
        if path:
            exportNpz(path, [self.laser])

    def onMenuExport(self):
        dlg = ExportDialog(
            self.laser.source,
            self.combo_isotope.currentText(),
            len(self.laser.isotopes()),
            self.laser.layers(),
            self,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        prompt_overwrite = True
        isotopes = self.laser.isotopes() if dlg.check_isotopes.isChecked() else [None]

        for isotope in isotopes:
            if dlg.check_layers.isChecked():
                for layer in range(self.laser.layers()):
                    path = dlg.getPath(isotope=isotope, layer=layer + 1)
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
            else:
                path = dlg.getPath(isotope=isotope)
                result = self._export(
                    path, isotope=isotope, layer=None, prompt_overwrite=prompt_overwrite
                )
                if result == QtWidgets.QMessageBox.No:
                    continue
                elif result == QtWidgets.QMessageBox.NoToAll:
                    return
                elif result == QtWidgets.QMessageBox.YesToAll:
                    prompt_overwrite = False

    def onMenuCalibration(self):
        pass

    def onMenuConfig(self):
        dlg = ConfigDialog(self.laser.config, parent=self)
        dlg.check_all.setEnabled(False)
        if dlg.exec() == ConfigDialog.Accepted:
            self.laser.config["spotsize"] = dlg.spotsize
            self.laser.config["speed"] = dlg.speed
            self.laser.config["scantime"] = dlg.scantime
            self.draw()

    def onMenuTrim(self):
        pass

    def onMenuClose(self):
        self.close()

    def onComboIsotope(self, text):
        self.draw()


class LaserImageDock(ImageDock):
    def __init__(self, laserdata, parent=None):

        super().__init__(parent)
        self.laser = laserdata
        self.combo_isotope.addItems(self.laser.isotopes())
        self.setWindowTitle(self.laser.name)

    def onMenuCalibration(self):
        dlg = CalibrationDialog(
            self.laser.calibration,
            self.combo_isotope.currentText(),
            self.laser.isotopes(),
            parent=self,
        )
        if dlg.exec():
            self.laser.calibration = dlg.calibration
            self.draw()

    def onMenuTrim(self):
        dlg = TrimDialog(self.laser.trimAs("s"), parent=self)
        dlg.check_all.setEnabled(False)
        if dlg.exec() == TrimDialog.Accepted:
            self.laser.setTrim(dlg.trim, dlg.combo_trim.currentText())
            self.draw()

    def _export(self, path, isotope=None, layer=None, prompt_overwrite=True):
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
    def __init__(self, kkdata, parent=None):

        super().__init__(kkdata, parent)
        self.setWindowTitle(f"kk:{self.laser.name}")

        # Config cannot be changed for krisskross images
        self.action_config.setEnabled(False)
        self.action_trim.setEnabled(False)

    def _export(self, path, isotope=None, layer=None, prompt_overwrite=True):
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
                :, :, layer
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
