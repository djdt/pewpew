import numpy as np

from pytestqt.qtbot import QtBot

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross

from pewpew.lib.viewoptions import ViewOptions
from pewpew.widgets.docks import LaserImageDock, KrissKrossImageDock


def test_laser_image_dock(qtbot: QtBot):
    dock = LaserImageDock(
        Laser.from_structured(
            np.array(np.random.random((10, 10)), dtype=[("B2", float), ("A1", float)])
        ),
        ViewOptions(),
    )
    qtbot.addWidget(dock)
    dock.buildContextMenu()
    dock.show()
    dock.combo_isotopes.setCurrentText("B2")
    dock.populateComboIsotopes()
    assert dock.combo_isotopes.currentText() == "A1"

    dlg = dock.onMenuCalibration()
    dlg.close()
    dlg = dock.onMenuConfig()
    dlg.close()
    dlg = dock.onMenuExport()
    dlg.close()
    dlg = dock.onMenuSave()
    dlg.close()
    dlg = dock.onMenuStats()
    dlg.close()

    dock.close()


def test_krisskross_image_dock(qtbot: QtBot):
    dock = KrissKrossImageDock(
        KrissKross.from_structured(
            [
                np.array(np.random.random((10, 60)), dtype=[("A1", float)]),
                np.array(np.random.random((10, 60)), dtype=[("A1", float)]),
            ]
        ),
        ViewOptions(),
    )
    qtbot.addWidget(dock)
    dock.buildContextMenu()
    dock.show()

    dlg = dock.onMenuConfig()
    dlg.close()
    dlg = dock.onMenuStats()
    dlg.close()

    dock.close()
