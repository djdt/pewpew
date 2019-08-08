import numpy as np

from pytestqt.qtbot import QtBot

from PySide2 import QtCore, QtWidgets

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross

from pewpew.main import MainWindow

from pewpew.widgets import dialogs, exporters
from pewpew.widgets.dialogs import ApplyDialog
from pewpew.widgets.docks import LaserImageDock, KrissKrossImageDock


def wait_for_and_close(dialog: type = ApplyDialog, max_time: int = 1000):
    if max_time < 0:
        QtWidgets.QApplication.exit()
        raise TimeoutError

    w = QtWidgets.QApplication.activeModalWidget()
    if isinstance(w, dialog):
        w.close()  # type: ignore
    else:
        QtCore.QTimer.singleShot(
            100, lambda: wait_for_and_close(dialog, max_time - 100)
        )


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
        MainWindow.DEFAULT_VIEW_CONFIG,
    )
    qtbot.addWidget(dock)
    dock.buildContextMenu()
    dock.show()
    dock.combo_isotope.setCurrentText("B2")
    dock.populateComboIsotopes()
    assert dock.combo_isotope.currentText() == "A1"

    qtbot.waitSignal

    wait_for_and_close(dialogs.CalibrationDialog)
    dock.onMenuCalibration()
    wait_for_and_close(dialogs.ConfigDialog)
    dock.onMenuConfig()
    wait_for_and_close(QtWidgets.QWidget)
    dock.onMenuExport()
    wait_for_and_close(QtWidgets.QWidget)
    dock.onMenuSave()
    wait_for_and_close(dialogs.StatsDialog)
    dock.onMenuStats()

    dock.close()


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
    dock.buildContextMenu()
    dock.show()

    wait_for_and_close(dialogs.ConfigDialog)
    dock.onMenuConfig()
    wait_for_and_close(dialogs.StatsDialog)
    dock.onMenuStats()
