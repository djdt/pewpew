import os
import copy

from PySide2 import QtCore, QtGui, QtWidgets

from laserlib import io
from laserlib.laser import Laser
from laserlib.config import LaserConfig
from laserlib.io.error import LaserLibException

from pewpew.lib.io import import_any
from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets import dialogs
from pewpew.widgets.canvases import InteractiveLaserCanvas
from pewpew.widgets.views import View, ViewSpace

from typing import Tuple


class LaserViewSpace(ViewSpace):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.config = LaserConfig()
        self.options = ViewOptions()

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

    def applyConfig(self, dlg: dialogs.ConfigDialog) -> None:
        if dlg.check_all.isChecked():
            self.config = copy.copy(dlg.config)
            for view in self.views:
                view.applyConfig(self.config)
        else:
            widget = self.activeView().activeWidget()
            widget.laser.config = copy.copy(dlg.config)
            widget.refresh()

    def applyCalibration(self, dlg: dialogs.CalibrationDialog) -> None:
        if dlg.check_all.isChecked():
            for view in self.views:
                view.applyCalibration(dlg.calibration)
        else:
            widget = self.activeView().activeWidget()
            for iso in widget.laser.isotopes:
                if iso in dlg.calibrations:
                    widget.laser.data[iso].calibration = copy.copy(
                        dlg.calibrations[iso]
                    )
            widget.refresh()


class LaserView(View):
    def refresh(self) -> None:
        if self.stack.count() > 0:
            self.stack.widget(self.stack.currentIndex()).refresh()

    def openDocument(self, paths: str, config: LaserConfig) -> None:
        try:
            for laser in import_any(paths, config):
                widget = LaserWidget(laser, self.viewspace.options)
                self.addTab(laser.name, widget)
        except LaserLibException as e:
            QtWidgets.QMessageBox.critical(self, type(e).__name__, f"{e}")
            return

    def saveDocument(self, path: str) -> bool:
        widget = self.activeWidget()
        io.npz.save(path, [widget.laser])
        widget.laser.filepath = path

    def applyConfig(self, config: LaserConfig) -> None:
        for widget in self.widgets():
            widget.laser.config = copy.copy(config)
        self.refresh()

    def applyCallibration(self, calibration: dict) -> None:
        for widget in self.widgets():
            for iso in widget.laser.isotopes:
                if iso in calibration:
                    widget.laser.data[iso].calibration = copy.copy(calibration[iso])
        self.refresh()


class LaserWidget(QtWidgets.QWidget):
    def __init__(
        self, laser: Laser, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.laser = laser

        self.canvas = InteractiveLaserCanvas(viewoptions)
        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.currentIndexChanged.connect(self.refresh)
        self.combo_isotopes.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
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
        layout_bar.addWidget(self.combo_isotopes, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        layout.addLayout(layout_bar)
        self.setLayout(layout)

    def laserFilePath(self, ext: str = ".npz") -> str:
        return os.path.join(os.path.dirname(self.laser.filepath), self.laser.name + ext)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = QtWidgets.QMenu(self)

        menu.addAction(self.window().action_open)
        menu.addAction(self.window().action_save)
        menu.addAction(self.window().action_export)
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
        self.canvas.drawLaser(self.laser, self.combo_isotopes.currentText())
