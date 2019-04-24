from PyQt5 import QtWidgets
import copy

from pewpew.ui.docks import LaserImageDock
from pewpew.ui.dialogs import ConfigDialog
from laserlib.krisskross import KrissKross

from pewpew.ui.dialogs import ApplyDialog


class KrissKrossImageDock(LaserImageDock):
    def __init__(self, laser: KrissKross, parent: QtWidgets.QWidget = None):

        super().__init__(laser, parent)
        self.setWindowTitle(f"kk:{self.laser.name}")

    def onMenuConfig(self) -> None:
        def applyDialog(dialog: ApplyDialog) -> None:
            if dialog.check_all.isChecked():
                docks = self.parent().findChildren(KrissKrossImageDock)
            else:
                docks = [self]
            for dock in docks:
                if type(dock.laser) == KrissKross:
                    dock.laser.config = copy.copy(dialog.config)
                    dock.draw()

        dlg = ConfigDialog(self.laser.config, parent=self)
        dlg.applyPressed.connect(applyDialog)
        if dlg.exec():
            applyDialog(dlg)
