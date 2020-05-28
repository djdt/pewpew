import numpy as np
import os.path
from pytestqt.qtbot import QtBot
from PySide2 import QtCore, QtGui

from pew.laser import Laser
from pew.config import Config
from pew.calibration import Calibration

from pewpew.widgets.laser import LaserViewSpace

from testing import rand_data


def test_laser_view_space(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()

    viewspace.splitActiveHorizontal()

    viewspace.views[0].addLaser(Laser(rand_data(["A1", "B2"])))
    viewspace.views[0].addLaser(Laser(rand_data(["A1", "C3"])))
    viewspace.views[1].addLaser(Laser(rand_data(["A1", "C3"])))
    viewspace.views[1].addLaser(Laser(rand_data(["B2", "D4"])))

    assert viewspace.uniqueIsotopes() == ["A1", "B2", "C3", "D4"]
    # Apply config
    viewspace.applyConfig(Config(10, 10, 10))
    for view in viewspace.views:
        for widget in view.widgets():
            assert widget.laser.config.spotsize == 10
            assert widget.laser.config.speed == 10
            assert widget.laser.config.scantime == 10
    # Try to apply calibraiton
    viewspace.applyCalibration(
        {"A1": Calibration(1.0, 1.0), "B2": Calibration(2.0, 2.0)}
    )
    qtbot.waitForWindowShown(viewspace)
    for view in viewspace.views:
        for widget in view.widgets():
            if "A1" in widget.laser.isotopes:
                assert widget.laser.calibration["A1"].intercept == 1.0
                assert widget.laser.calibration["A1"].gradient == 1.0
            if "B2" in widget.laser.isotopes:
                assert widget.laser.calibration["B2"].intercept == 2.0
                assert widget.laser.calibration["B2"].gradient == 2.0
            if "C3" in widget.laser.isotopes:
                assert widget.laser.calibration["C3"].intercept == 0.0
                assert widget.laser.calibration["C3"].gradient == 1.0

    # Check isotope changed if avilable
    assert viewspace.views[0].activeWidget().combo_isotope.currentText() == "A1"
    assert viewspace.views[1].activeWidget().combo_isotope.currentText() == "A1"
    viewspace.setCurrentIsotope("B2")
    assert viewspace.views[0].activeWidget().combo_isotope.currentText() == "B2"
    assert viewspace.views[1].activeWidget().combo_isotope.currentText() == "A1"
    # Close all
    for view in viewspace.views:
        for widget in view.widgets():
            widget.close()


def test_laser_view(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    laser = view.addLaser(Laser(rand_data(["A1", "B2", "C3"])))
    qtbot.waitForWindowShown(laser)

    view.tabs.setTabText(0, "newname")
    assert view.stack.widget(0).laser.name == "newname"

    view.contextMenuEvent(
        QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0))
    )

    # Drop event
    drag_mime = QtCore.QMimeData()
    path = os.path.join(os.path.dirname(__file__), "data", "io", "npz.npz")
    drag_mime.setUrls([QtCore.QUrl.fromLocalFile(path)])
    drag_event = QtGui.QDragEnterEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.CopyAction,
        drag_mime,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    view.dragEnterEvent(drag_event)
    assert drag_event.isAccepted()
    drop_event = QtGui.QDropEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.CopyAction,
        drag_mime,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    view.dropEvent(drop_event)
    assert len(view.widgets()) == 2

    dlg = view.actionOpen()
    dlg.show()
    dlg.close()


def test_laser_widget(qtbot: QtBot):
    x = rand_data(["A1", "B2"])
    y = x["A1"].copy()
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()

    view = viewspace.activeView()
    view.addLaser(Laser(x))

    widget = view.activeWidget()

    widget.applyConfig(Config(1.0, 1.0, 1.0))
    assert widget.laser.config.spotsize == 1.0
    widget.applyCalibration({"B2": Calibration(2.0, 2.0)})
    assert widget.laser.calibration["B2"].intercept == 2.0

    widget.crop((1.5, 9.5, 1.5, 9.5))
    assert view.activeWidget().laser.data.shape == (8, 8)
    widget.transform(flip="horizontal")
    assert np.all(widget.laser.get("A1") == np.flip(y, axis=1))
    widget.transform(flip="horizontal")
    widget.transform(flip="vertical")
    assert np.all(widget.laser.get("A1") == np.flip(y, axis=0))
    widget.transform(flip="vertical")
    assert np.all(widget.laser.get("A1") == y)
    widget.transform(rotate="right")
    assert np.all(widget.laser.get("A1") == np.rot90(y, k=1, axes=(1, 0)))
    widget.transform(rotate="left")
    assert np.all(widget.laser.get("A1") == y)


def test_laser_widget_actions(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    view.addLaser(Laser(rand_data(["a", "b"])))
    widget = view.activeWidget()

    dlg = widget.actionCalibration()
    dlg.close()
    dlg = widget.actionConfig()
    dlg.close()
    widget.actionCopyImage()
    dlg = widget.actionExport()
    dlg.close()
    dlg = widget.actionSave()
    dlg.close()
    dlg = widget.actionStatistics()
    dlg.close()
    dlg = widget.actionSelectDialog()
    dlg.close()
    dlg = widget.actionColocal()
    dlg.close()

    widget.contextMenuEvent(
        QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0))
    )
