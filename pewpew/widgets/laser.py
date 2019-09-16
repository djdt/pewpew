import os
import copy

from PySide2 import QtCore, QtGui, QtWidgets

from laserlib import io
from laserlib.laser import Laser
from laserlib.krisskross import KrissKross, KrissKrossConfig
from laserlib.config import LaserConfig
from laserlib.io.error import LaserLibException

from pewpew.lib.io import import_any
from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import InteractiveLaserCanvas
from pewpew.widgets import dialogs, exportdialogs
from pewpew.widgets.views import View, ViewSpace

from typing import Callable, List


def qAction(icon: str, label: str, status: str, func: Callable) -> QtWidgets.QAction:
    action = QtWidgets.QAction(QtGui.QIcon.fromTheme(icon), label)
    action.setStatusTip(status)
    action.triggered.connect(func)
    return action


class LaserViewSpace(ViewSpace):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.config = LaserConfig()
        self.options = ViewOptions()

        # self.action_config = qAction(
        #     "document-edit", "Config", "Edit the documents config.", self.actionConfig
        # )
        # self.action_calibration = qAction(
        #     "go-top",
        #     "Calibration",
        #     "Edit the documents calibration.",
        #     self.actionCalibration,
        # )

    def uniqueIsotopes(self) -> List[str]:
        isotopes = set()
        for view in self.views:
            for widget in view.widgets():
                isotopes.update(widget.laser.isotopes)
        return sorted(isotopes)

    def createView(self) -> "LaserView":
        view = LaserView(self)
        view.numTabsChanged.connect(self.numTabsChanged)
        self.views.append(view)
        self.numViewsChanged.emit()
        return view

    def openDocument(self, paths: str) -> None:
        view = self.activeView()
        view.openDocument(paths, self.config)

    def saveDocument(self, path: str) -> None:
        view = self.activeView()
        view.saveDocument(path)

    def setCurrentIsotope(self, isotope: str) -> None:
        for view in self.views:
            view.setCurrentIsotope(isotope)

    def applyConfig(self, config: LaserConfig) -> None:
        self.config = copy.copy(config)
        for view in self.views:
            view.applyConfig(self.config)
        self.refresh()

    def applyCalibration(self, calibration: dict) -> None:
        for view in self.views:
            view.applyCalibration(calibration)
        self.refresh()

    @QtCore.Slot("QWidget*")
    def mouseSelectStart(self, callback_widget: QtWidgets.QWidget) -> None:
        for view in self.views:
            for widget in view.widgets():
                widget.canvas.installEventFilter(callback_widget)

    @QtCore.Slot("QWidget*")
    def mouseSelectEnd(self, callback_widget: QtWidgets.QWidget) -> None:
        for view in self.views:
            for widget in view.widgets():
                widget.canvas.removeEventFilter(callback_widget)


class LaserView(View):
    def __init__(self, viewspace: ViewSpace, parent: QtWidgets.QWidget = None):
        super().__init__(viewspace, parent)
        self.tabs.tabTextChanged.connect(self.renameLaser)

    def addLaser(self, laser: Laser) -> int:
        widget = LaserWidget(laser, self.viewspace.options, self)
        name = laser.name if laser.name != "" else "__noname__"
        return self.addTab(name, widget)

    def renameLaser(self, index: int) -> None:
        self.stack.widget(index).laser.name = self.tabs.tabText(index)
        self.setTabModified(index)

    def setTabModified(self, index: int, modified: bool = True) -> None:
        if modified:
            self.tabs.setTabIcon(index, QtGui.QIcon.fromTheme("document-save"))
        else:
            self.tabs.setTabIcon(index, QtGui.QIcon())

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.window().action_open)
        menu.popup(event.globalPos())

    def refresh(self) -> None:
        if self.stack.count() > 0:
            self.stack.widget(self.stack.currentIndex()).refresh()

    def openDocument(self, paths: str, config: LaserConfig) -> None:
        try:
            for laser in import_any(paths, config):
                self.addLaser(laser)
        except LaserLibException as e:
            QtWidgets.QMessageBox.critical(self, type(e).__name__, f"{e}")

    def saveDocument(self, path: str) -> bool:
        widget = self.activeWidget()
        io.npz.save(path, [widget.laser])
        widget.laser.filepath = path
        self.setTabModified(self.stack.indexOf(widget), False)

    def applyConfig(self, config: LaserConfig) -> None:
        for index, widget in enumerate(self.widgets()):
            if not isinstance(config, KrissKrossConfig) or widget.is_srr:
                widget.laser.config = copy.copy(config)
            else:  # Manually fill in the 3
                widget.laser.config.spotsize = config.spotsize
                widget.laser.config.speed = config.speed
                widget.laser.config.scantime = config.scantime
            self.setTabModified(index)
        self.refresh()

    def applyCalibration(self, calibration: dict) -> None:
        for index, widget in enumerate(self.widgets()):
            modified = False
            for iso in widget.laser.isotopes:
                if iso in calibration:
                    widget.laser.data[iso].calibration = copy.copy(calibration[iso])
                    modified = True
            if modified:
                self.setTabModified(index)
        self.refresh()

    def setCurrentIsotope(self, isotope: str) -> None:
        for widget in self.widgets():
            if isotope in widget.laser.isotopes:
                widget.combo_isotopes.setCurrentText(isotope)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if not event.mimeData().hasUrls():
            return super().dropEvent(event)

        paths = [
            url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()
        ]
        try:
            lasers = import_any(paths, self.viewspace.config)
            for laser in lasers:
                self.addLaser(laser)
            event.acceptProposedAction()
        except io.error.LaserLibException:
            event.ignore()


class LaserWidget(QtWidgets.QWidget):
    laserModified = QtCore.Signal()

    def __init__(
        self,
        laser: Laser,
        viewoptions: ViewOptions,
        view: LaserView,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.view = view
        self.laser = laser
        self.is_srr = isinstance(laser, KrissKross)

        self.canvas = InteractiveLaserCanvas(viewoptions, parent=self)
        # self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.combo_layers = QtWidgets.QComboBox()
        self.combo_layers.addItem("*")
        self.combo_layers.addItems([str(i) for i in range(0, self.laser.layers)])
        self.combo_layers.currentIndexChanged.connect(self.refresh)
        if not self.is_srr:
            self.combo_layers.setEnabled(False)
            self.combo_layers.setVisible(False)

        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo_isotopes.currentIndexChanged.connect(self.refresh)
        self.populateIsotopes()

        self.view_button = QtWidgets.QToolButton()
        self.view_button.setAutoRaise(True)
        self.view_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.view_button.setIcon(QtGui.QIcon.fromTheme("zoom-in"))
        self.view_button.addAction(QtWidgets.QAction("zo"))
        self.view_button.installEventFilter(self)

        layout_bar = QtWidgets.QHBoxLayout()
        layout_bar.addWidget(self.view_button, 0, QtCore.Qt.AlignLeft)
        layout_bar.addStretch(1)
        layout_bar.addWidget(self.combo_layers, 0, QtCore.Qt.AlignRight)
        layout_bar.addWidget(self.combo_isotopes, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        layout.addLayout(layout_bar)
        self.setLayout(layout)

    def laserFilePath(self, ext: str = ".npz") -> str:
        return os.path.join(os.path.dirname(self.laser.filepath), self.laser.name + ext)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = QtWidgets.QMenu(self)

        menu.addAction(self.action_copy_image)
        menu.addSeparator()
        menu.addAction(self.action_open)
        menu.addAction(self.action_save)
        menu.addAction(self.action_export)
        menu.addSeparator()
        menu.addAction(self.action_config)
        menu.addAction(self.action_calibration)
        menu.addAction(self.action_statistics)
        menu.exec_(event.globalPos())

    def populateIsotopes(self) -> None:
        self.combo_isotopes.blockSignals(True)
        self.combo_isotopes.clear()
        self.combo_isotopes.addItems(self.laser.isotopes)
        self.combo_isotopes.blockSignals(False)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        self.refresh()
        super().showEvent(event)

    def refresh(self) -> None:
        if self.combo_layers.currentIndex() == 0:
            layer = None
        else:
            layer = int(self.combo_layers.currentText())

        self.canvas.drawLaser(
            self.laser, self.combo_isotopes.currentText(), layer=layer
        )

    def applyConfig(self, config: LaserConfig) -> None:
        if not isinstance(config, KrissKrossConfig) or self.is_srr:
            self.laser.config = copy.copy(config)
        else:  # Manually fill in the 3
            self.laser.config.spotsize = config.spotsize
            self.laser.config.speed = config.speed
            self.laser.config.scantime = config.scantime
        self.view.setTabModified(self.view.stack.indexOf(self))
        self.refresh()

    def applyCalibration(self, calibrations: dict) -> None:
        for iso in self.laser.isotopes:
            if iso in calibrations:
                self.laser.data[iso].calibration = copy.copy(calibrations[iso])
        self.view.setTabModified(self.view.stack.indexOf(self))
        self.refresh()

    # Actions
    def actionCopyImage(self) -> None:
        self.canvas.copyToClipboard()

    def actionConfig(self) -> QtWidgets.QDialog:
        dlg = dialogs.ConfigDialog(self.laser.config, parent=self)
        dlg.configSelected.connect(
            lambda c, b: self.view.viewspace.applyConfig(c)
            if b
            else self.applyConfig(c)
        )
        dlg.open()
        return dlg

    def actionCalibration(self) -> QtWidgets.QDialog:
        calibrations = {k: self.laser.data[k].calibration for k in self.laser.data}
        dlg = dialogs.CalibrationDialog(
            calibrations, self.combo_isotopes.currentText(), parent=self
        )
        dlg.calibrationSelected.connect(
            lambda c, b: self.view.viewspace.applyCalibration(c)
            if b
            else self.applyCalibration(c)
        )
        dlg.open()
        return dlg

    def actionExport(self) -> QtWidgets.QDialog:
        dlg = exportdialogs.ExportDialog(
            self.laser,
            self.combo_isotopes.currentText(),
            self.canvas.view_limits,
            self.canvas.viewoptions,
            self,
        )
        dlg.open()
        return dlg


    def actionSave(self) -> QtWidgets.QDialog:
        filepath = self.laser.filepath
        if filepath.lower().endswith(".npz") and os.path.exists(filepath):
            self.viewspace.saveDocument(filepath)
            return None
        else:
            filepath = self.laserFilePath()
        dlg = QtWidgets.QFileDialog(
            self, "Save File", filepath, "Numpy archive(*.npz);;All files(*)"
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.fileSelected.connect(self.viewspace.saveDocument)
        dlg.open()
        return dlg

    def actionStatistics(self) -> QtWidgets.QDialog:
        data = self.canvas.getMaskedData()
        area = (
            self.laser.config.get_pixel_width() * self.laser.config.get_pixel_height()
        )
        dlg = dialogs.StatsDialog(data, area, self.canvas.image.get_clim(), parent=self)
        dlg.open()
        return dlg
