import numpy as np

from pytestqt.qtbot import QtBot

from laserlib.laser import Laser

from pewpew.main import MainWindow
from pewpew.widgets.docks import LaserImageDock


def test_main_window_empty(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert not window.action_calibration.isEnabled()
    assert not window.action_export.isEnabled()
    assert not window.action_operations.isEnabled()

    dlg = window.menuOpen()
    dlg.close()
    dlg = window.menuImportAgilent()
    dlg.close()
    dlg = window.menuImportThermoiCap()
    dlg.close()
    dlg = window.menuImportKrissKross()
    dlg.close()
    dlg = window.menuConfig()
    dlg.close()
    dlg = window.menuColormapRange()
    dlg.close()


def test_main_window_laser(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.dockarea.addDockWidgets(
        [
            LaserImageDock(
                Laser.from_structured(
                    np.array(np.random.random((10, 10)), dtype=[("A1", float)])
                ),
                window.viewoptions,
            )
        ]
    )

    assert window.action_calibration.isEnabled()
    assert window.action_export.isEnabled()
    assert window.action_operations.isEnabled()

    dlg = window.menuExportAll()
    dlg.close()
    dlg = window.menuStandardsTool()
    dlg.close()
    dlg = window.menuOperationsTool()
    dlg.close()
