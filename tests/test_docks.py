from pytestqt.qtbot import QtBot
from PySide2 import QtCore, QtWidgets
import numpy as np

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross

from pewpew.ui.mainwindow import MainWindow

from pewpew.ui.docks.dockarea import DockArea
from pewpew.ui.docks.laserdock import LaserImageDock
from pewpew.ui.docks.krisskrossdock import KrissKrossImageDock

from pewpew.ui.dialogs import ApplyDialog, CalibrationDialog


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


# def test_dock_area(qtbot: QtBot):
#     dockarea = DockArea()
#     qtbot.addWidget(dockarea)
#     dockarea.show()

#     with qtbot.waitSignal(dockarea.numberDocksChanged):
#         dockarea.addDockWidgets(widgets)
#     assert dockarea.visibleDocks() == 2
