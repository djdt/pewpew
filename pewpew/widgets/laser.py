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
from pewpew.widgets.views import View, ViewSpace

from typing import List


class LaserViewSpace(ViewSpace):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.config = LaserConfig()
        self.options = ViewOptions()

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

    def applyConfig(self, config: LaserConfig) -> None:
        self.config = copy.copy(config)
        for view in self.views:
            view.applyConfig(self.config)
        self.refresh()

    def applyCalibration(self, calibration: dict) -> None:
        for view in self.views:
            view.applyCalibration(calibration)
        self.refresh()

    def setCurrentIsotope(self, isotope: str) -> None:
        for view in self.views:
            view.setCurrentIsotope(isotope)

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
        widget = LaserWidget(laser, self.viewspace.options)
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
        menu.exec_(event.globalPos())

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
        for widget in self.widgets():
            if not isinstance(config, KrissKrossConfig) or widget.is_srr:
                widget.laser.config = copy.copy(config)
            else:  # Manually fill in the 3
                widget.laser.config.spotsize = config.spotsize
                widget.laser.config.speed = config.speed
                widget.laser.config.scantime = config.scantime
        self.refresh()

    def applyCalibration(self, calibration: dict) -> None:
        for widget in self.widgets():
            for iso in widget.laser.isotopes:
                if iso in calibration:
                    widget.laser.data[iso].calibration = copy.copy(calibration[iso])
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
        self, laser: Laser, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
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

        menu.addAction(self.window().action_copy_image)
        menu.addSeparator()
        menu.addAction(self.window().action_open)
        menu.addAction(self.window().action_save)
        menu.addAction(self.window().action_export)
        menu.addSeparator()
        menu.addAction(self.window().action_config)
        menu.addAction(self.window().action_calibration)
        menu.addAction(self.window().action_statistics)
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
