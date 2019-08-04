from pytestqt.qtbot import QtBot
from PySide2 import QtWidgets
import numpy as np

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross

from pewpew.ui.mainwindow import MainWindow

from pewpew.ui.docks.dockarea import DockArea
from pewpew.ui.docks.laserdock import LaserImageDock
from pewpew.ui.docks.krisskrossdock import KrissKrossImageDock


def test_laser_image_dock(qtbot: QtBot):
    dock = LaserImageDock(
        Laser.from_structured(
            np.array(np.random.random((10, 10)), dtype=[("A1", float)])
        ),
        MainWindow.DEFAULT_VIEW_CONFIG,
    )
    qtbot.addWidget(dock)
    dock.draw()
    dock.show()


def test_krisskross_image_dock(qtbot: QtBot):
    dock = KrissKrossImageDock(
        KrissKross.from_structured(
            [
                np.array(np.random.random((10, 100)), dtype=[("A1", float)]),
                np.array(np.random.random((10, 100)), dtype=[("A1", float)]),
            ]
        ),
        MainWindow.DEFAULT_VIEW_CONFIG,
    )
    qtbot.addWidget(dock)
    dock.draw()
    dock.show()


# def test_dock_area(qtbot: QtBot):
#     dockarea = DockArea()
#     qtbot.addWidget(dockarea)
#     dockarea.show()

#     with qtbot.waitSignal(dockarea.numberDocksChanged):
#         dockarea.addDockWidgets(widgets)
#     assert dockarea.visibleDocks() == 2
