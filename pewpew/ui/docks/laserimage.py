from PyQt5 import QtWidgets

from pewpew.ui.docks import ImageDock

from pewpew.lib.laser import LaserData


class LaserImageDock(ImageDock):
    def __init__(self, laserdata: LaserData, parent: QtWidgets.QWidget = None):

        super().__init__(parent)
        self.laser = laserdata
        self.combo_isotope.addItems(self.laser.isotopes())
        self.setWindowTitle(self.laser.name)
