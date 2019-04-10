from PyQt5 import QtWidgets

from pewpew.ui.docks import LaserImageDock

from pewpew.lib.laser.krisskross import KrissKross


class KrissKrossImageDock(LaserImageDock):
    def __init__(self, kkdata: KrissKross, parent: QtWidgets.QWidget = None):

        super().__init__(kkdata, parent)
        self.setWindowTitle(f"kk:{self.laser.name}")
        # Config cannot be changed for krisskross images
        self.action_config.setEnabled(False)

    def onMenuConfig(self) -> None:
        pass
