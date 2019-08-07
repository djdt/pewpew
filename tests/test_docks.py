import numpy as np

from pytestqt.qtbot import QtBot

from PySide2 import QtCore, QtWidgets

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross

from pewpew.main import MainWindow

from pewpew.widgets.dialogs import ApplyDialog
from pewpew.widgets.docks import LaserImageDock, KrissKrossImageDock


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
    dock.buildContextMenu().actions()
    dock.show()
    dock.combo_isotope.setCurrentText("B2")
    dock.populateComboIsotopes()
    assert dock.combo_isotope.currentText() == "A1"

    QtCore.QTimer.singleShot(500, close_active_modal)
    dock.onMenuCalibration()
    assert not has_dialog(ApplyDialog)
    QtCore.QTimer.singleShot(500, close_active_modal)
    dock.onMenuConfig()
    assert not has_dialog(ApplyDialog)
    QtCore.QTimer.singleShot(500, close_active_modal)
    dock.onMenuExport()
    assert not has_dialog(ApplyDialog)
    QtCore.QTimer.singleShot(500, close_active_modal)
    dock.onMenuSave()
    assert not has_dialog(ApplyDialog)
    QtCore.QTimer.singleShot(1000, close_active_modal)
    dock.onMenuStats()
    assert not has_dialog(ApplyDialog)


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
