from PyQt5 import QtCore, QtWidgets
import copy

from pewpew.ui.docks import LaserImageDock
from pewpew.ui.dialogs import ConfigDialog
from laserlib.krisskross import KrissKross

from pewpew.ui.dialogs import ApplyDialog


class KrissKrossImageDock(LaserImageDock):
    def __init__(self, laser: KrissKross, parent: QtWidgets.QWidget = None):
        super().__init__(laser, parent)

        self.combo_layer = QtWidgets.QComboBox()
        self.combo_layer.currentIndexChanged.connect(self.onComboLayer)
        self.combo_layer.addItem("*")
        self.combo_layer.addItems([str(i) for i in range(0, self.laser.layers())])
        self.layout_bottom.insertWidget(0, self.combo_layer, 1, QtCore.Qt.AlignRight)

        # self.setWindowTitle(f"kk:{self.laser.name}")

    def draw(self) -> None:
        try:
            layer = int(self.combo_layer.currentText())
        except ValueError:
            layer = None

        self.canvas.drawLaser(self.laser, self.combo_isotope.currentText(), layer=layer)
        self.canvas.draw()

    def onComboLayer(self, text: str) -> None:
        self.draw()

    def onMenuConfig(self) -> None:
        def applyDialog(dialog: ApplyDialog) -> None:
            if dialog.check_all.isChecked():
                docks = self.parent().findChildren(KrissKrossImageDock)
            else:
                docks = [self]
            for dock in docks:
                if type(dock.laser) == KrissKross:
                    if not dock.laser.check_config_valid(dialog.config):
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Invalid config!",
                            "Config is not valid for current image(s).",
                        )
                        return
                    dock.laser.config = copy.copy(dialog.config)
                    dock.draw()

        dlg = ConfigDialog(self.laser.config, parent=self)
        dlg.applyPressed.connect(applyDialog)
        if dlg.exec():
            applyDialog(dlg)
