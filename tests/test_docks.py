import numpy as np

from pytestqt.qtbot import QtBot

from PySide2 import QtWidgets

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross

from pewpew.main import MainWindow

from pewpew.widgets import dialogs
from pewpew.widgets.docks import LaserImageDock, KrissKrossImageDock

from funcs import wait_for_and_close_modal, wait_for_and_close_top_level


def test_laser_image_dock(qtbot: QtBot):
    dock = LaserImageDock(
        Laser.from_structured(
            np.array(np.random.random((10, 10)), dtype=[("B2", float), ("A1", float)])
        ),
        MainWindow.DEFAULT_VIEW_CONFIG,
    )
    qtbot.addWidget(dock)
    dock.buildContextMenu()
    dock.show()
    dock.combo_isotope.setCurrentText("B2")
    dock.populateComboIsotopes()
    assert dock.combo_isotope.currentText() == "A1"

    wait_for_and_close_modal(dialogs.CalibrationDialog)
    dock.onMenuCalibration()
    wait_for_and_close_modal(dialogs.ConfigDialog)
    dock.onMenuConfig()
    wait_for_and_close_modal(QtWidgets.QWidget)
    dock.onMenuExport()
    wait_for_and_close_modal(QtWidgets.QWidget)
    dock.onMenuSave()
    wait_for_and_close_top_level(dialogs.StatsDialog)
    dock.onMenuStats()

    dock.close()


def test_krisskross_image_dock(qtbot: QtBot):
    dock = KrissKrossImageDock(
        KrissKross.from_structured(
            [
                np.array(np.random.random((10, 60)), dtype=[("A1", float)]),
                np.array(np.random.random((10, 60)), dtype=[("A1", float)]),
            ]
        ),
        MainWindow.DEFAULT_VIEW_CONFIG,
    )
    qtbot.addWidget(dock)
    dock.buildContextMenu()
    dock.show()

    wait_for_and_close_modal(dialogs.ConfigDialog)
    dock.onMenuConfig()
    wait_for_and_close_top_level(dialogs.StatsDialog)
    dock.onMenuStats()

    dock.close()
