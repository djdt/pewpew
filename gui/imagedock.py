from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from util.laserimage import plotLaserImage
from util.plothelpers import coords2value
from util.exporter import exportCsv, exportNpz, exportPng, exportVtr
from util.formatter import isotopeFormat

from gui.dialogs import (CalibrationDialog, ConfigDialog, ExportDialog,
                         TrimDialog)

import numpy as np
import os.path


# TODO, draw calls in config will reset cmap
class Canvas(FigureCanvasQTAgg):
    def __init__(self, fig, parent=None):
        super().__init__(fig)
        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

        self.rubber_band = None
        self.rubber_band_origin = QtCore.QSize()

    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self):
        return QtCore.QSize(250, 250)


class ImageDock(QtWidgets.QDockWidget):
    DEFAULT_VIEW_CONFIG = {
        'cmap': 'magma',
        'interpolation': 'none',
        'cmap_range': ('2%', '98%'),
        'fontsize': 10,
    }

    def __init__(self, parent=None):

        self.laser = None
        self.image = np.array([])

        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable
                         | QtWidgets.QDockWidget.DockWidgetMovable)

        self.fig = Figure(
            frameon=False, tight_layout=True, figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = Canvas(self.fig, self)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.onComboIsotope)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.combo_isotope, 1, QtCore.Qt.AlignRight)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

        # Context menu actions
        self.action_copy = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('edit-copy'), "Open Copy", self)
        self.action_copy.setStatusTip("Open a copy of this data")
        self.action_copy.triggered.connect(self.onMenuCopy)
        self.action_save = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('document-save'), "Save", self)
        self.action_save.setStatusTip("Save data to archive.")
        self.action_save.triggered.connect(self.onMenuSave)

        self.action_export = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('document-save-as'), "Export", self)
        self.action_export.setStatusTip("Export data to different formats.")
        self.action_export.triggered.connect(self.onMenuExport)

        self.action_calibration = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('go-top'), "Calibration", self)
        self.action_calibration.setStatusTip("Edit image calibration.")
        self.action_calibration.triggered.connect(self.onMenuCalibration)

        self.action_config = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('document-properties'), "Config", self)
        self.action_config.setStatusTip("Edit image config.")
        self.action_config.triggered.connect(self.onMenuConfig)

        self.action_trim = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('edit-cut'), "Trim", self)
        self.action_trim.setStatusTip("Edit image trim.")
        self.action_trim.triggered.connect(self.onMenuTrim)

        self.action_close = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('edit-delete'), "Close", self)
        self.action_close.setStatusTip("Close the images.")
        self.action_close.triggered.connect(self.onMenuClose)

        # Canvas actions
        self.canvas.mpl_connect('motion_notify_event', self.updateStatusBar)
        self.canvas.mpl_connect('axes_leave_event', self.clearStatusBar)

    def updateStatusBar(self, e):
        if e.inaxes == self.ax:
            x, y = e.xdata, e.ydata
            v = coords2value(self.image, x, y)
            self.window().statusBar().showMessage(f"{x:.2f},{y:.2f} [{v}]")

    def clearStatusBar(self, e):
        self.window().statusBar().clearMessage()

    def draw(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        isotope = self.combo_isotope.currentText()
        viewconfig = self.window().viewconfig

        self.image = plotLaserImage(
            self.fig,
            self.ax,
            self.laser.get(isotope, calibrated=True, trimmed=True),
            colorbar='bottom',
            colorbarlabel=self.laser.calibration['units'].get(isotope, ""),
            label=isotopeFormat(isotope),
            fontsize=viewconfig['fontsize'],
            cmap=viewconfig['cmap'],
            interpolation=viewconfig['interpolation'],
            vmin=viewconfig['cmap_range'][0],
            vmax=viewconfig['cmap_range'][1],
            aspect=self.laser.aspect(),
            extent=self.laser.extent(trimmed=True))

        self.canvas.draw()

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
            self, "Save", "", "Numpy archive(*.npz);;All files(*)")
        if path:
            exportNpz(path, [self.laser])

    def onMenuExport(self):
        dlg = ExportDialog(self.laser.source, self.combo_isotope.currentText(),
                           len(self.laser.isotopes()), self.laser.layers(),
                           self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        prompt_overwrite = True
        isotopes = self.laser.isotopes() \
            if dlg.check_isotopes.isChecked() else [None]

        for isotope in isotopes:
            if dlg.check_layers.isChecked():
                for layer in range(self.laser.layers()):
                    path = dlg.getPath(isotope=isotope, layer=layer + 1)
                    result = self._export(
                        path,
                        isotope=isotope,
                        layer=layer,
                        prompt_overwrite=prompt_overwrite)
                    if result == QtWidgets.QMessageBox.No:
                        continue
                    elif result == QtWidgets.QMessageBox.NoToAll:
                        return
                    elif result == QtWidgets.QMessageBox.YesToAll:
                        prompt_overwrite = False
            else:
                path = dlg.getPath(isotope=isotope)
                result = self._export(
                    path,
                    isotope=isotope,
                    layer=None,
                    prompt_overwrite=prompt_overwrite)
                if result == QtWidgets.QMessageBox.No:
                    continue
                elif result == QtWidgets.QMessageBox.NoToAll:
                    return
                elif result == QtWidgets.QMessageBox.YesToAll:
                    prompt_overwrite = False

    def onMenuCalibration(self):
        dlg = CalibrationDialog(
            self.laser.calibration,
            self.combo_isotope.currentText(),
            self.laser.isotopes(),
            parent=self)
        if dlg.exec():
            self.laser.calibration = dlg.calibration
            self.draw()

    def onMenuConfig(self):
        dlg = ConfigDialog(self.laser.config, parent=self)
        dlg.check_all.setEnabled(False)
        if dlg.exec() == ConfigDialog.Accepted:
            self.laser.config['spotsize'] = dlg.spotsize
            self.laser.config['speed'] = dlg.speed
            self.laser.config['scantime'] = dlg.scantime
            self.draw()

    def onMenuTrim(self):
        dlg = TrimDialog(self.laser.trimAs('s'), parent=self)
        dlg.check_all.setEnabled(False)
        if dlg.exec() == TrimDialog.Accepted:
            self.laser.setTrim(dlg.trim, dlg.combo_trim.currentText())
            self.draw()

    def onMenuClose(self):
        self.close()

    def onComboIsotope(self, text):
        self.draw()


class LaserImageDock(ImageDock):
    def __init__(self, laserdata, parent=None):

        super().__init__(parent)
        self.laser = laserdata
        self.combo_isotope.addItems(self.laser.isotopes())
        name = os.path.splitext(os.path.basename(self.laser.source))[0]
        self.setWindowTitle(name)

    def _export(self, path, isotope=None, layer=None, prompt_overwrite=True):
        if isotope is None:
            isotope = self.combo_isotope.currentText()

        result = QtWidgets.QMessageBox.Yes
        if prompt_overwrite and os.path.exists(path):
            result = QtWidgets.QMessageBox.warning(
                self, "Overwrite File?",
                f"The file \"{os.path.basename(path)}\" "
                "already exists. Do you wish to overwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.YesToAll
                | QtWidgets.QMessageBox.No)
            if result == QtWidgets.QMessageBox.No:
                return result

        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            exportCsv(path,
                      self.laser.get(isotope, calibrated=True, trimmed=True),
                      isotope, self.laser.config)
        elif ext == '.npz':
            exportNpz(path, [self.laser])
        elif ext == '.png':
            exportPng(path,
                      self.laser.get(isotope, calibrated=True, trimmed=True),
                      isotope, self.laser.aspect(),
                      self.laser.extent(trimmed=True),
                      self.window().viewconfig)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Format",
                f"Unknown extention for \'{os.path.basename(path)}\'.")
            return QtWidgets.QMessageBox.NoToAll

        return result


class KrissKrossImageDock(ImageDock):
    def __init__(self, kkdata, parent=None):

        super().__init__(parent)
        self.laser = kkdata
        self.combo_isotope.addItems(self.laser.isotopes())
        name = os.path.splitext(os.path.basename(self.laser.source))[0]
        self.setWindowTitle(f"{name}:kk")

        # Config cannot be changed for krisskross images
        self.action_config.setEnabled(False)
        self.action_trim.setEnabled(False)

    def onMenuCopy(self):
        dock_copy = type(self)(self.laser, self.parent())
        dock_copy.draw()
        # Split by longest length
        size = self.size()
        self.parent().splitDockWidget(
            self, dock_copy,
            QtCore.Qt.Horizontal if size[0] > size[1] else QtCore.Qt.Vertical)

    def _export(self, path, isotope=None, layer=None, prompt_overwrite=True):
        if isotope is None:
            isotope = self.combo_isotope.currentText()

        result = QtWidgets.QMessageBox.Yes
        if prompt_overwrite and os.path.exists(path):
            result = QtWidgets.QMessageBox.warning(
                self, "Overwrite File?",
                f"The file \"{os.path.basename(path)}\" "
                "already exists. Do you wish to overwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.YesToAll
                | QtWidgets.QMessageBox.No)
            if result == QtWidgets.QMessageBox.No:
                return result

        if layer is None:
            export_data = self.laser.get(
                isotope, calibrated=True, flattened=True)
        else:
            export_data = self.laser.get(
                isotope, calibrated=True, flattened=False)[:, :, layer]

        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            np.savetxt(path, export_data, delimiter=',')
            exportCsv(path, export_data, isotope, self.laser.config)
        elif ext == '.npz':
            exportNpz(path, [self.laser])
        elif ext == '.png':
            exportPng(path, export_data, isotope, self.laser.aspect(),
                      self.laser.extent(trimmed=True),
                      self.window().viewconfig)
        elif ext == '.vtr':
            exportVtr(path, self.laser)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Format",
                f"Unknown extention for \"{os.path.basename(path)}\".")
            return QtWidgets.QMessageBox.NoToAll

        return result
