import numpy as np

from pytestqt.qtbot import QtBot

from laserlib.laser import Laser
from laserlib.config import LaserConfig
from laserlib.calibration import LaserCalibration

from pewpew.widgets.laser import LaserWidget, LaserView, LaserViewSpace


def test_laser_view_space(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()

    viewspace.splitActiveHorizontal()

    for i in range(1, 6):
        x = np.array(
            np.random.random([10, 10]), dtype=[(str(i), float), (str(i + 1), float)]
        )
        laser = Laser.from_structured(
            x, name="laser" + str(i), filepath=f"/home/laser{i}.npz"
        )
        viewspace.views[i % 2].addLaser(laser)

    assert viewspace.uniqueIsotopes() == ["1", "2", "3", "4", "5", "6"]

    viewspace.applyConfig(LaserConfig(10, 10, 10))
    for view in viewspace.views:
        for widget in view.widgets():
            assert widget.laser.config.spotsize == 10
            assert widget.laser.config.speed == 10
            assert widget.laser.config.scantime == 10

    viewspace.applyCalibration(
        {"1": LaserCalibration(1.0, 1.0), "2": LaserCalibration(2.0, 2.0)}
    )
    for view in viewspace.views:
        for widget in view.widgets():
            if "1" in widget.laser.isotopes:
                assert widget.laser.data["1"].calibration.intercept == 1.0
                assert widget.laser.data["1"].calibration.gradient == 1.0
            if "2" in widget.laser.isotopes:
                assert widget.laser.data["2"].calibration.intercept == 2.0
                assert widget.laser.data["2"].calibration.gradient == 2.0

    assert viewspace.activeWidget().combo_isotopes.currentText() == "2"
    viewspace.setCurrentIsotope("3")
    assert viewspace.activeWidget().combo_isotopes.currentText() == "3"


def test_laser_view(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    view = viewspace.activeView()
