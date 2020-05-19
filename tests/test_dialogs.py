import numpy as np

from pytestqt.qtbot import QtBot

from PySide2 import QtCore, QtGui

from pew.config import Config
from pew.calibration import Calibration
from pew.srr.config import SRRConfig

from pewpew.lib.viewoptions import ViewOptions
from pewpew.widgets.dialogs import (
    ApplyDialog,
    CalibrationDialog,
    CalibrationCurveDialog,
    ColorRangeDialog,
    ConfigDialog,
    StatsDialog,
)


def test_apply_dialog(qtbot: QtBot):
    dialog = ApplyDialog()
    qtbot.addWidget(dialog)
    dialog.open()

    for button in dialog.button_box.buttons():
        dialog.buttonClicked(button)


def test_calibration_dialog(qtbot: QtBot):
    cals = {
        "A": Calibration.from_points([[0, 1], [1, 2]]),
        "B": Calibration(),
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
    dialog.check_all.setChecked(True)
    dialog.apply()


def test_calibration_curve_dialog(qtbot: QtBot):
    dialog = CalibrationCurveDialog(
        Calibration.from_points([[0, 1], [1, 2], [2, 3], [4, 4]])
    )
    qtbot.addWidget(dialog)
    dialog.open()

    dialog.contextMenuEvent(
        QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0))
    )


def test_colorrange_dialog(qtbot: QtBot):
    viewoptions = ViewOptions()
    viewoptions.colors.default_range = (0.0, 1.0)
    viewoptions.colors._ranges = {"A": (1.0, 2.0), "B": ("2%", 3.0)}
    dialog = ColorRangeDialog(viewoptions, ["A", "B", "C"], "C")
    qtbot.addWidget(dialog)
    dialog.open()

    # Loads C as current, has default range
    assert dialog.combo_isotopes.currentText() == "C"
    assert dialog.lineedit_min.text() == ""
    assert dialog.lineedit_max.text() == ""
    assert dialog.lineedit_min.placeholderText() == "0.0"
    assert dialog.lineedit_max.placeholderText() == "1.0"
    # Not added yet
    assert "C" not in dialog.ranges
    # Add and check is there
    dialog.lineedit_min.setText("1%")
    dialog.lineedit_max.setText("2%")
    dialog.combo_isotopes.setCurrentText("B")  # Update C
    assert dialog.ranges["C"] == ("1%", "2%")

    assert dialog.lineedit_min.text() == "2%"
    assert dialog.lineedit_max.text() == "3.0"

    dialog.combo_isotopes.setCurrentText("A")
    assert dialog.lineedit_min.text() == "1.0"
    assert dialog.lineedit_max.text() == "2.0"

    dialog.check_all.click()
    dialog.lineedit_min.setText("1.0")
    dialog.lineedit_max.setText("2.0")
    # dialog.combo_isotopes.setCurrentText("C")

    dialog.apply()

    assert dialog.default_range == (1.0, 2.0)
    assert dialog.ranges == {}


def test_laser_config_dialog(qtbot: QtBot):
    config = Config()
    dialog = ConfigDialog(config)
    qtbot.addWidget(dialog)
    dialog.open()

    assert not hasattr(dialog, "lineedit_warmup")
    assert not hasattr(dialog, "spinbox_offsets")
    # Check the texts are correct
    assert dialog.lineedit_spotsize.text() == str(config.spotsize)
    assert dialog.lineedit_speed.text() == str(config.speed)
    assert dialog.lineedit_scantime.text() == str(config.scantime)
    dialog.lineedit_spotsize.setText("1")
    dialog.lineedit_speed.setText("2.")
    dialog.lineedit_scantime.setText("3.0000")
    dialog.updateConfig()
    # Check it updated
    assert dialog.config.spotsize == 1.0
    assert dialog.config.speed == 2.0
    assert dialog.config.scantime == 3.0

    dialog.apply()
    dialog.check_all.setChecked(True)
    dialog.apply()


def test_config_dialog_krisskross(qtbot: QtBot):
    dialog = ConfigDialog(SRRConfig())
    qtbot.addWidget(dialog)
    dialog.open()

    assert hasattr(dialog, "lineedit_warmup")
    assert hasattr(dialog, "spinbox_offsets")

    qtbot.mouseClick(dialog.spinbox_offsets, QtCore.Qt.LeftButton)
    dialog.lineedit_warmup.setText("7.5")
    dialog.spinbox_offsets.setValue(3)
    dialog.updateConfig()

    assert dialog.config.warmup == 7.5
    assert dialog.config._subpixel_size == 3

    dialog.apply()
    dialog.check_all.setChecked(True)
    dialog.apply()


def test_stats_dialog(qtbot: QtBot):
    x = np.array(np.random.random([10, 10]), dtype=[('a', float)])
    x[0, 0] = np.nan
    m = np.full(x.shape, True, dtype=bool)

    dialog = StatsDialog(x, m, 'a', (0, 1))
    qtbot.addWidget(dialog)
    dialog.open()

    dialog.contextMenuEvent(
        QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0))
    )
