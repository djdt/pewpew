import numpy as np

from pytestqt.qtbot import QtBot

from laserlib.laser import Laser

from pewpew.main import MainWindow
from pewpew.widgets.docks import LaserImageDock


def test_main_window_dialogs(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)

    dlg = window.menuOpen()
    dlg.close()
    dlg = window.menuImportAgilent()
    dlg.close()
    dlg = window.menuImportThermoiCap()
    dlg.close()
    dlg = window.menuImportKrissKross()
    dlg.close()
    dlg = window.menuColormapRange()
    dlg.close()
    dlg = window.menuExportAll()
    assert dlg is None
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
    dlg = window.menuExportAll()
    dlg.close()
