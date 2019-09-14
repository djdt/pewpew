import numpy as np

from pytestqt.qtbot import QtBot
from PySide2 import QtCore, QtGui

from laserlib.laser import Laser
from laserlib.config import LaserConfig
from laserlib.calibration import LaserCalibration

from pewpew.widgets.laser import LaserWidget, LaserView, LaserViewSpace

from typing import List


def rand_laser(names: List[str]) -> Laser:
    dtype = [(name, float) for name in names]
    return Laser.from_structured(
        np.array(np.random.random((10, 10)), dtype=dtype),
        name="laser",
        filepath="/home/laser.npz",
    )


def test_laser_view_space(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()

    viewspace.splitActiveHorizontal()

    viewspace.views[0].addLaser(rand_laser(["A1", "B2"]))
    viewspace.views[0].addLaser(rand_laser(["A1", "C3"]))
    viewspace.views[1].addLaser(rand_laser(["A1", "C3"]))
    viewspace.views[1].addLaser(rand_laser(["B2", "D4"]))

    assert viewspace.uniqueIsotopes() == ["A1", "B2", "C3", "D4"]
    # Apply config
    viewspace.applyConfig(LaserConfig(10, 10, 10))
    for view in viewspace.views:
        for widget in view.widgets():
            assert widget.laser.config.spotsize == 10
            assert widget.laser.config.speed == 10
            assert widget.laser.config.scantime == 10
    # Try to apply calibraiton
    viewspace.applyCalibration(
        {"A1": LaserCalibration(1.0, 1.0), "B2": LaserCalibration(2.0, 2.0)}
    )
    qtbot.waitForWindowShown(viewspace)
    for view in viewspace.views:
        for widget in view.widgets():
            if "A1" in widget.laser.isotopes:
                assert widget.laser.data["A1"].calibration.intercept == 1.0
                assert widget.laser.data["A1"].calibration.gradient == 1.0
            if "B2" in widget.laser.isotopes:
                assert widget.laser.data["B2"].calibration.intercept == 2.0
                assert widget.laser.data["B2"].calibration.gradient == 2.0
            if "C3" in widget.laser.isotopes:
                assert widget.laser.data["C3"].calibration.intercept == 0.0
                assert widget.laser.data["C3"].calibration.gradient == 1.0

    # Check isotope changed if avilable
    assert viewspace.views[0].activeWidget().combo_isotopes.currentText() == "A1"
    assert viewspace.views[1].activeWidget().combo_isotopes.currentText() == "A1"
    viewspace.setCurrentIsotope("B2")
    assert viewspace.views[0].activeWidget().combo_isotopes.currentText() == "B2"
    assert viewspace.views[1].activeWidget().combo_isotopes.currentText() == "A1"
    # Close all
    for view in viewspace.views:
        for widget in view.widgets():
            widget.close()


def test_laser_view(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    view = viewspace.activeView()

    view.addLaser(rand_laser(["A1", "B2", "C3"]))

    view.tabs.setTabText(0, "newname")
    assert view.stack.widget(0).laser.name == "newname"

    view.contextMenuEvent(
        QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0))
    )
