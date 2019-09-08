from PySide2 import QtCore, QtGui, QtWidgets

from laserlib.laser import Laser

from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.actions import ActionGenerator
from pewpew.widgets.canvases import InteractiveLaserCanvas
from pewpew.widgets.views import View, ViewTitleBar


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

    def contextMenu(self) -> QtWidgets.QMenu:
        context_menu = QtWidgets.QMenu(self)
        context_menu.addAction(
            ActionGenerator().getAction("copy-image", self.canvas.copyToClipboard)
        )
        context_menu.addSeparator()
        context_menu.addAction(ActionGenerator().getAction("save"), self.menuSave)
        context_menu.addAction(ActionGenerator().getAction("export"), self.menuExport)
        context_menu.addSeparator()
        context_menu.addAction(
            ActionGenerator().getAction("dialog-calibration"), self.menuCalibration
        )
        context_menu.addAction(
            ActionGenerator().getAction("dialog-config"), self.menuConfig
        )
        context_menu.addAction(
            ActionGenerator().getAction("dialog-statistics"), self.menuStats
        )
        return context_menu

    def populateIsotopes(self) -> None:
        self.combo_isotopes.blockSignals(True)
        self.combo_isotopes.clear()
        self.combo_isotopes.addItems(self.laser.isotopes)
        self.combo_isotopes.blockSignals(False)

    def refresh(self) -> None:
        self.canvas.drawLaser(self.laser, self.combo_isotopes.currentText())

    def menuSave(self) -> QtWidgets.QDialog:
        def save_npz(path: str):
            io.npz.save(path, [self.laser])
            self.laser.filepath = path

        if self.laser.filepath.lower().endswith(".npz") and os.path.exists(
            self.laser.filepath
        ):
            save_npz(self.laser.filepath)
            return None

        dlg = QtWidgets.QFileDialog(
            self,
            "Save File",
            os.path.join(
                os.path.dirname(self.laser.filepath), self.laser.name + ".npz"
            ),
            "Numpy archive(*.npz);;All files(*)",
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.fileSelected.connect(save_npz)
        dlg.open()
        return dlg

    def menuExport(self) -> QtWidgets.QDialog:
        dlg = ExportDialog(
            self.laser,
            self.combo_isotopes.currentText(),
            self.canvas.view_limits,
            self.canvas.viewoptions,
            self,
        )
        dlg.open()
        return dlg

    def menuCalibration(self) -> QtWidgets.QDialog:
        def applyDialog(dialog: dialogs.ApplyDialog) -> None:
            if dialog.check_all.isChecked():
                docks = self.window().findChildren(LaserWidget)
            else:
                docks = [self]
            for dock in docks:
                for isotope in dlg.calibrations.keys():
                    if isotope in dock.laser.isotopes:
                        dock.laser.data[isotope].calibration = copy.copy(
                            dlg.calibrations[isotope]
                        )
                dock.draw()

        calibrations = {
            isotope: self.laser.data[isotope].calibration
            for isotope in self.laser.data.keys()
        }
        dlg = dialogs.CalibrationDialog(
            calibrations, self.combo_isotopes.currentText(), parent=self
        )
        dlg.applyPressed.connect(applyDialog)
        dlg.open()
        return dlg

    def menuConfig(self) -> QtWidgets.QDialog:
        def applyDialog(dialog: dialogs.ApplyDialog) -> None:
            if dialog.check_all.isChecked():
                docks = self.window().findChildren(LaserWidget)
            else:
                docks = [self]
            for dock in docks:
                if type(dock.laser) == Laser:
                    dock.laser.config = copy.copy(dialog.config)
                    dock.draw()

        dlg = dialogs.ConfigDialog(self.laser.config, parent=self)
        dlg.applyPressed.connect(applyDialog)
        dlg.open()
        return dlg

    def menuStats(self) -> QtWidgets.QDialog:
        data = self.canvas.image.get_array()
        mask = self.canvas.getSelection()
        if mask is not None and not np.all(mask == 0):
            # Trim out nan rows and columns to get size
            data = np.where(mask, data, np.nan)
            data = data[:, ~np.isnan(data).all(axis=0)]
            data = data[~np.isnan(data).all(axis=1)]
        else:  # Trim to view limits
            x0, x1, y0, y1 = self.canvas.view_limits
            (x0, y0), (x1, y1) = (
                image_extent_to_data(self.canvas.image)
                .transform(((x0, y1), (x1, y0)))
                .astype(int)
            )
            data = data[y0:y1, x0:x1]  # type: ignore

        dlg = dialogs.StatsDialog(
            data,
            self.canvas.viewoptions.colors.get_range_as_float(
                self.combo_isotopes.currentText(), data
            ),
            parent=self,
        )
        dlg.open()
        return dlg
