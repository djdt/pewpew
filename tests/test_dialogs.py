import numpy as np

from pytestqt.qtbot import QtBot

from PySide2 import QtCore

from laserlib.config import LaserConfig
from laserlib.calibration import LaserCalibration
from laserlib.krisskross.config import KrissKrossConfig

from pewpew.lib.viewoptions import ViewOptions
from pewpew.widgets.dialogs import (
    ApplyDialog,
    CalibrationDialog,
    CalibrationCurveDialog,
    ColorRangeDialog,
    ConfigDialog,
    MultipleDirDialog,
    StatsDialog,
)


def test_apply_dialog(qtbot: QtBot):
    dialog = ApplyDialog()
    qtbot.addWidget(dialog)
    dialog.open()

    for button in dialog.button_box.buttons():
        dialog.buttonClicked(button)

    dialog.close()

def test_calibration_dialog(qtbot: QtBot):
    cals = {
        "A": LaserCalibration.from_points([[0, 1], [1, 2]]),
        "B": LaserCalibration(),
    }
    dialog = CalibrationDialog(cals, "B")
    qtbot.addWidget(dialog)
    dialog.open()

    assert dialog.combo_isotopes.currentText() == "B"
    assert not dialog.button_plot.isEnabled()

    dialog.lineedit_gradient.setText("1")
    dialog.lineedit_intercept.setText("2")
    dialog.lineedit_unit.setText("ppm")

    dialog.combo_isotopes.setCurrentIndex(0)
    assert dialog.combo_isotopes.currentText() == "A"
    assert dialog.button_plot.isEnabled()

    assert dialog.calibrations["B"].gradient == 1.0
    assert dialog.calibrations["B"].intercept == 2.0
    assert dialog.calibrations["B"].unit == "ppm"

    dialog.showCurve()

    dialog.apply()
    dialog.close()


def test_calibration_curve_dialog(qtbot: QtBot):
    dialog = CalibrationCurveDialog(
        LaserCalibration.from_points([[0, 1], [1, 2], [2, 3], [4, 4]])
    )
    qtbot.addWidget(dialog)
    dialog.open()


def test_colorrange_dialog(qtbot: QtBot):
    dialog = ColorRangeDialog(ViewOptions(), ["A", "B", "C"])
    qtbot.addWidget(dialog)
    dialog.open()

    dialog.lineedit_min.setText("1%")
    dialog.lineedit_max.setText("999.9")
    dialog.combo_isotopes.setCurrentText("B")
    assert dialog.ranges["A"] == ("1%", 999.9)
    assert "B" not in dialog.ranges
    dialog.combo_isotopes.setCurrentText("A")
    assert dialog.lineedit_min.text() == "1%"
    assert dialog.lineedit_max.text() == "999.9"
    assert "B" not in dialog.ranges

    dialog.apply()
    dialog.close()


def test_laser_config_dialog(qtbot: QtBot):
    config = LaserConfig()
    dialog = ConfigDialog(config)
    qtbot.addWidget(dialog)
    dialog.open()

    assert not hasattr(dialog, "lineedit_warmup")
    assert not hasattr(dialog, "spinbox_offsets")
    # Check the texts are correct
    assert dialog.lineedit_spotsize.placeholderText() == str(config.spotsize)
    assert dialog.lineedit_speed.placeholderText() == str(config.speed)
    assert dialog.lineedit_scantime.placeholderText() == str(config.scantime)
    dialog.lineedit_spotsize.setText("1")
    dialog.lineedit_speed.setText("2.")
    dialog.lineedit_scantime.setText("3.0000")
    dialog.updateConfig()
    # Check it updated
    assert dialog.config.spotsize == 1.0
    assert dialog.config.speed == 2.0
    assert dialog.config.scantime == 3.0

    dialog.apply()
    dialog.close()


def test_config_dialog_krisskross(qtbot: QtBot):
    dialog = ConfigDialog(KrissKrossConfig())
    qtbot.addWidget(dialog)
    dialog.open()

    assert hasattr(dialog, "lineedit_warmup")
    assert hasattr(dialog, "spinbox_offsets")

    qtbot.mouseClick(dialog.spinbox_offsets, QtCore.Qt.LeftButton)
    dialog.lineedit_warmup.setText("7.5")
    dialog.spinbox_offsets.setValue(3)
    dialog.updateConfig()

    assert dialog.config.warmup == 7.5  # type: ignore
    assert dialog.config._subpixel_size == 3  # type: ignore

    dialog.apply()
    dialog.close()


def test_multi_dir_dialog(qtbot: QtBot):
    dialog = MultipleDirDialog(None, "MDD", "")
    qtbot.addWidget(dialog)
    dialog.open()
    dialog.close()


def test_stats_dialog(qtbot: QtBot):
    x = np.random.random([10, 10])
    x[0, 0] = np.nan

    dialog = StatsDialog(x, 10, (0, 1))
    qtbot.addWidget(dialog)
    dialog.open()
    dialog.close()
