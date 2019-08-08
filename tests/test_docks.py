import numpy as np

from pytestqt.qtbot import QtBot
from pytestqt.exceptions import TimeoutError

from PySide2 import QtCore, QtWidgets

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross

from pewpew.main import PPMainWindow

from pewpew.widgets import dialogs
from dialogs import ApplyDialog
from pewpew.widgets.docks import LaserImageDock, KrissKrossImageDock


def wait_for_and_close(dialog: type = ApplyDialog, max_time: int = 1000):
    if max_time < 0:
        raise TimeoutError
    w = QtWidgets.QApplication.activeModalWidget()
    if isinstance(w, dialog):
        w.close()
        return
    QtCore.QTimer.singleShot(100, lambda: wait_for_and_close(dialog, max_time - 100))


def close_active_modal():
    w = QtWidgets.QApplication.activeModalWidget()
    assert w is not None and w.isVisible()
    w.close()


def has_dialog(dialog: type = ApplyDialog):
    widgets = QtWidgets.QApplication.topLevelWidgets()
    assert any(isinstance(w, dialog) for w in widgets)


def test_laser_image_dock(qtbot: QtBot):
    dock = LaserImageDock(
        Laser.from_structured(
            np.array(np.random.random((10, 10)), dtype=[("B2", float), ("A1", float)])
        ),
        PPMainWindow.DEFAULT_VIEW_CONFIG,
    )
    qtbot.addWidget(dock)
    dock.buildContextMenu().actions()
    dock.show()
    dock.combo_isotope.setCurrentText("B2")
    dock.populateComboIsotopes()
    assert dock.combo_isotope.currentText() == "A1"

    wait_for_and_close(ApplyDialog)
    dock.onMenuCalibration()
    wait_for_and_close(ApplyDialog)
    dock.onMenuConfig()
    wait_for_and_close(ApplyDialog)
    dock.onMenuExport()
    wait_for_and_close(ApplyDialog)
    dock.onMenuSave()
    wait_for_and_close(ApplyDialog)
    dock.onMenuStats()


def test_krisskross_image_dock(qtbot: QtBot):
    dock = KrissKrossImageDock(
        KrissKross.from_structured(
            [
                np.array(np.random.random((10, 100)), dtype=[("A1", float)]),
                np.array(np.random.random((10, 100)), dtype=[("A1", float)]),
            ]
        ),
        PPMainWindow.DEFAULT_VIEW_CONFIG,
    )
    qtbot.addWidget(dock)
    dock.draw()
    dock.show()
