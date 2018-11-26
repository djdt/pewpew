from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from util.laserimage import plotLaserImage
from util.plothelpers import coords2value
from util.exporter import exportNpz

from gui.qt.configdialog import ConfigDialog

import numpy as np
import os.path

# TODO, draw calls in config will reset cmap


class ImageDock(QtWidgets.QDockWidget):
    DEFAULT_VIEW_CONFIG = {'cmap': 'magma',
                           'interpolation': 'none',
                           'cmap_range': ('1%', '99%')}

    def __init__(self, parent=None):

        self.laser = np.array([])
        self.image = np.array([])

        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                         QtWidgets.QDockWidget.DockWidgetMovable)

        self.fig = Figure(frameon=False, figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.canvas.setMinimumSize(100, 100)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                  QtWidgets.QSizePolicy.MinimumExpanding)

        self.setWidget(self.canvas)

        # Context menu actions
        self.action_save = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('document-save'), "Save", self)
        self.action_save.setStatusTip("Save image to archive.")
        self.action_save.triggered.connect(self.onMenuSave)

        self.action_save_as = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('document-save-as'), "Save As", self)
        self.action_save_as.setStatusTip("Save image to a different format.")
        self.action_save_as.triggered.connect(self.onMenuSaveAs)

        self.action_config = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('document-properties'), "Config", self)
        self.action_config.setStatusTip("Edit image config.")
        self.action_config.triggered.connect(self.onMenuConfig)

        self.action_close = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('edit-delete'), "Close", self)
        self.action_close.setStatusTip("Close the image.")
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

    def draw(self, viewconfig):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        self.image = plotLaserImage(
            self.fig, self.ax, self.laser.calibrated(),
            colorbar='bottom', label=self.laser.isotope,
            cmap=viewconfig['cmap'], interpolation=viewconfig['interpolation'],
            aspect=self.laser.aspect(), extent=self.laser.extent())

        self.fig.tight_layout()
        self.canvas.draw()

    def contextMenuEvent(self, event):
        context_menu = QtWidgets.QMenu(self)
        context_menu.addAction(self.action_save)
        context_menu.addAction(self.action_save_as)
        context_menu.addAction(self.action_config)
        context_menu.addSeparator()
        context_menu.addAction(self.action_close)
        context_menu.exec(event.globalPos())

    def onMenuSave(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save", "",
                "Numpy archive(*.npz);;All files(*)")
        if path:
            exportNpz(path, [self.laser])

    def onMenuSaveAs(self):
        pass

    def onMenuConfig(self):
        dlg = ConfigDialog(self.laser.config, parent=self)
        dlg.check_all.setEnabled(False)
        if dlg.exec():
            self.laser.config = dlg.form.config
            self.draw()

    def onMenuClose(self):
        self.close()


class LaserImageDock(ImageDock):
    def __init__(self, laserdata, parent=None):

        super().__init__(parent)
        self.laser = laserdata
        name = os.path.splitext(os.path.basename(self.laser.source))[0]
        self.setWindowTitle(f"{name}:{self.laser.isotope}")

    def draw(self, viewconfig, laserdata=None):
        if laserdata is not None:
            self.laser = laserdata
        super().draw(viewconfig)

    def onMenuSaveAs(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save As", "", "CSV files(*.csv);;"
                "Numpy archives(*.npz);;PNG images(*.png);;All files(*)")
        if path:
            ext = os.path.splitext(path)[1]
            if ext == ".npz":
                exportNpz(path, [self.laser])
            elif ext == ".csv":
                np.savetxt(path, self.laser.calibrated(), delimiter=',')
            elif ext == ".png":
                # TODO show a config dialog
                pass
            else:
                QtWidgets.QMessageBox.warning(
                        self, "Invalid Format",
                        f"Unknown extention for \'{os.path.basename(path)}\'.")


class KrissKrossImageDock(ImageDock):
    def __init__(self, kkdata, parent=None):

        super().__init__(parent)
        self.laser = kkdata
        name = os.path.splitext(os.path.basename(self.laser.source))[0]
        self.setWindowTitle(f"{name}:kk:{self.laser.isotope}")

        self.action_export = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('document-send'), "Export layers", self)
        self.action_export.setStatusTip("Export layers to individual files.")
        self.action_export.triggered.connect(self.onMenuExport)

    def draw(self, viewconfig, kkdata=None):
        if kkdata is not None:
            self.laser = kkdata
        super().draw(viewconfig)

    def contextMenuEvent(self, event):
        context_menu = QtWidgets.QMenu(self)
        context_menu.addAction(self.action_save)
        context_menu.addAction(self.action_save_as)
        context_menu.addAction(self.action_export)
        context_menu.addAction(self.action_config)
        context_menu.addSeparator()
        context_menu.addAction(self.action_close)
        context_menu.exec(event.globalPos())

    def onMenuSaveAs(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save As", "", "CSV files(*.csv);;"
                "Numpy archives(*.npz);;PNG images(*.png);;"
                "Rectilinear VTK(*.vtr);;All files(*)")
        if path:
            ext = os.path.splitext(path)[1]
            if ext == ".npz":
                exportNpz(path, [self.laser])
            elif ext == ".csv":
                np.savetxt(path, self.laser.calibrated(), delimiter=',')
            # elif ext == ".vtr":
            # elif ext == ".png":
            #     # TODO show a config dialog
            #     pass
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Format",
                    f"Unknown extention for \'{os.path.basename(path)}\'.")

    def onMenuExport(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Enter File Basename", "", "Numpy archives(*.npz);;"
                "CSV files(*.csv);;PNG images(*.png);;All files(*)", "",
                QtWidgets.QFileDialog.DontConfirmOverwrite)
        if path:
            base, ext = os.path.splitext(path)
            yes_to_all = False
            for i, ld in enumerate(self.laser.split(), 1):
                # Check for existing files and prompt for overwrite
                layer_path = f"{base}_layer{i}{ext}"
                if os.path.exists(layer_path) and not yes_to_all:
                    result = QtWidgets.QMessageBox.warning(
                        self, "Overwrite File?",
                        f"The file \"{os.path.basename(layer_path)}\" "
                        "already exists. Do you wish to overwrite it?",
                        QtWidgets.QMessageBox.Yes |
                        QtWidgets.QMessageBox.YesToAll |
                        QtWidgets.QMessageBox.No)
                    if result == QtWidgets.QMessageBox.No:
                        continue
                    elif result == QtWidgets.QMessageBox.YesToAll:
                        yes_to_all = True
                if ext == ".npz":
                        exportNpz(layer_path, [ld])
                elif ext == ".csv":
                        np.savetxt(layer_path, ld.data, delimiter=',')
                else:
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Format",
                        f"Unknown extention for \'{os.path.basename(path)}\'.")
                    break
