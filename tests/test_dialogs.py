import numpy as np

from pytestqt.qtbot import QtBot

from PySide2 import QtCore, QtGui

from pewlib.config import Config
from pewlib.calibration import Calibration
from pewlib.srr.config import SRRConfig

from pewpew.lib.viewoptions import ViewOptions
from pewpew.widgets import dialogs
from pewpew.widgets.canvases import SelectableImageCanvas


def test_apply_dialog(qtbot: QtBot):
    dialog = dialogs.ApplyDialog()
    qtbot.addWidget(dialog)
    dialog.open()

    for button in dialog.button_box.buttons():
        dialog.buttonClicked(button)


def test_calibration_dialog(qtbot: QtBot):
    cals = {
        "A": Calibration.from_points([[0, 2], [1, 4]], unit="ppb"),
        "B": Calibration(),
    }

    dialog = dialogs.CalibrationDialog(cals, "B")
    qtbot.addWidget(dialog)
    dialog.open()

    assert dialog.combo_isotope.currentText() == "B"
    assert not dialog.button_plot.isEnabled()

    dialog.lineedit_gradient.setText("1")
    dialog.lineedit_intercept.setText("2")
    dialog.lineedit_unit.setText("ppm")

    dialog.combo_isotope.setCurrentIndex(0)
    assert dialog.combo_isotope.currentText() == "A"
    assert dialog.button_plot.isEnabled()

    assert dialog.calibrations["B"].gradient == 1.0
    assert dialog.calibrations["B"].intercept == 2.0
    assert dialog.calibrations["B"].unit == "ppm"

    dialog.showCurve()

    dialog.apply()
    dialog.check_all.setChecked(True)
    dialog.apply()


def test_calibration_curve_dialog(qtbot: QtBot):
    dialog = dialogs.CalibrationCurveDialog(
        Calibration.from_points([[0, 1], [1, 2], [2, 3], [4, 4]])
    )
    qtbot.addWidget(dialog)
    dialog.open()

    dialog.contextMenuEvent(
        QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0))
    )


def test_colocalisation_dialog(qtbot: QtBot):
    data = np.empty((10, 10), dtype=[("a", float), ("b", float), ("c", float)])
    data["a"] = np.repeat(np.linspace(0, 1, 10).reshape(1, -1), 10, axis=0)
    data["b"] = np.repeat(np.linspace(0, 1, 10).reshape(-1, 1), 10, axis=1)
    np.random.seed(9764915)
    data["c"] = np.random.random((10, 10))

    mask = np.ones((10, 10), dtype=bool)
    mask[:2] = False

    dialog = dialogs.ColocalisationDialog(data, mask)
    qtbot.addWidget(dialog)
    dialog.open()

    assert dialog.combo_name1.currentText() == "a"
    assert dialog.combo_name2.currentText() == "b"

    assert dialog.label_r.text() == "0.00"
    assert dialog.label_icq.text() == "0.00"
    assert dialog.label_m1.text() == "0.00"
    assert dialog.label_m2.text() == "0.50"
    assert dialog.label_p.text() == ""

    dialog.combo_name2.setCurrentText("a")

    assert dialog.label_r.text() == "1.00"
    assert dialog.label_icq.text() == "0.50"
    assert dialog.label_m1.text() == "1.00"
    assert dialog.label_m2.text() == "1.00"
    assert dialog.label_p.text() == ""

    dialog.calculatePearsonsProbablity()
    assert dialog.label_p.text() == "1.00"


def test_colorrange_dialog(qtbot: QtBot):
    viewoptions = ViewOptions()
    viewoptions.colors.default_range = (0.0, 1.0)
    viewoptions.colors._ranges = {"A": (1.0, 2.0), "B": ("2%", 3.0)}

    dialog = dialogs.ColorRangeDialog(viewoptions, ["A", "B", "C"], "C")
    qtbot.addWidget(dialog)
    dialog.open()

    # Loads C as current, has default range
    assert dialog.combo_isotope.currentText() == "C"
    assert dialog.lineedit_min.text() == ""
    assert dialog.lineedit_max.text() == ""
    assert dialog.lineedit_min.placeholderText() == "0.0"
    assert dialog.lineedit_max.placeholderText() == "1.0"
    # Not added yet
    assert "C" not in dialog.ranges
    # Add and check is there
    dialog.lineedit_min.setText("1%")
    dialog.lineedit_max.setText("2%")
    dialog.combo_isotope.setCurrentText("B")  # Update C
    assert dialog.ranges["C"] == ("1%", "2%")

    assert dialog.lineedit_min.text() == "2%"
    assert dialog.lineedit_max.text() == "3.0"

    dialog.combo_isotope.setCurrentText("A")
    assert dialog.lineedit_min.text() == "1.0"
    assert dialog.lineedit_max.text() == "2.0"

    dialog.check_all.click()
    dialog.lineedit_min.setText("1.0")
    dialog.lineedit_max.setText("2.0")
    # dialog.combo_isotope.setCurrentText("C")

    dialog.apply()

    assert dialog.default_range == (1.0, 2.0)
    assert dialog.ranges == {}


def test_laser_config_dialog(qtbot: QtBot):
    config = Config()

    dialog = dialogs.ConfigDialog(config)
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

    assert dialog.isComplete()
    dialog.lineedit_scantime.setText("-1")
    assert not dialog.isComplete()
    dialog.lineedit_speed.setText("-1")
    assert not dialog.isComplete()
    dialog.lineedit_spotsize.setText("-1")
    assert not dialog.isComplete()

    dialog.apply()
    dialog.check_all.setChecked(True)
    dialog.apply()


def test_config_dialog_krisskross(qtbot: QtBot):
    dialog = dialogs.ConfigDialog(SRRConfig())
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

    assert dialog.isComplete()
    dialog.lineedit_warmup.setText("-1")
    assert not dialog.isComplete()

    dialog.apply()
    dialog.check_all.setChecked(True)
    dialog.apply()


def test_name_edit_dialog(qtbot: QtBot):
    dialog = dialogs.NameEditDialog(["a", "b"], allow_remove=True)
    qtbot.addWidget(dialog)
    dialog.open()

    dialog.addName("c")
    dialog.addNames(["d", "e"])

    dialog.list.item(0).setText("A")
    dialog.list.item(1).setCheckState(QtCore.Qt.Unchecked)

    assert np.all(
        [
            dialog.list.item(i).data(dialog.originalNameRole)
            for i in range(dialog.list.count())
        ]
        == ["a", "b", "c", "d", "e"],
    )
    assert np.all(
        [dialog.list.item(i).text() for i in range(dialog.list.count())]
        == ["A", "b", "c", "d", "e"],
    )

    with qtbot.waitSignal(dialog.namesSelected) as emit:
        dialog.accept()

    assert np.all(list(emit.args[0].keys()) == ["a", "c", "d", "e"])


def test_selection_dialog(qtbot: QtBot):
    x = np.random.random((10, 10))
    canvas = SelectableImageCanvas()
    canvas.drawFigure()
    canvas.image = canvas.ax.imshow(x)

    dialog = dialogs.SelectionDialog(canvas)
    qtbot.addWidget(dialog)
    dialog.open()

    # Test enabling of options
    assert dialog.lineedit_manual.isEnabled()
    assert not dialog.spinbox_method.isEnabled()
    assert not dialog.spinbox_comparison.isEnabled()

    dialog.combo_method.setCurrentText("K-means")
    dialog.refresh()
    assert not dialog.lineedit_manual.isEnabled()
    assert dialog.spinbox_method.isEnabled()
    assert dialog.spinbox_comparison.isEnabled()
    assert dialog.spinbox_method.value() == 2
    assert dialog.spinbox_comparison.value() == 1

    dialog.combo_method.setCurrentText("Mean")
    dialog.refresh()
    assert not dialog.spinbox_method.isEnabled()
    assert not dialog.spinbox_comparison.isEnabled()

    # Test correct states and masks emmited
    with qtbot.wait_signal(dialog.maskSelected) as emitted:
        dialog.apply()
    assert np.all(emitted.args[0] == (x > x.mean()))
    assert emitted.args[1] == ""

    dialog.check_limit_selection.setChecked(True)
    dialog.combo_method.setCurrentText("Manual")
    dialog.lineedit_manual.setText("0.9")
    dialog.refresh()

    with qtbot.wait_signal(dialog.maskSelected) as emitted:
        dialog.apply()
    assert np.all(emitted.args[0] == (x > 0.9))
    assert emitted.args[1] == "intersect"

    dialog.canvas.selection = emitted.args[0]

    # Test limit threshold
    dialog.combo_method.setCurrentText("Mean")
    dialog.check_limit_selection.setChecked(False)
    dialog.check_limit_threshold.setChecked(True)
    dialog.refresh()

    with qtbot.wait_signal(dialog.maskSelected) as emitted:
        dialog.apply()
    assert np.all(emitted.args[0] == (x > np.mean(x[x > 0.9])))


def test_stats_dialog(qtbot: QtBot):
    x = np.array(np.random.random([10, 10]), dtype=[("a", float)])
    x[0, :] = np.nan
    m = np.full(x.shape, True, dtype=bool)

    dialog = dialogs.StatsDialog(x, m, "a", (1.0, 1.0))
    qtbot.addWidget(dialog)
    dialog.open()

    assert dialog.label_area.text().endswith("90 μm²")
    dialog.pixel_size = (1e3, 1e3)
    dialog.updateStats()
    assert dialog.label_area.text().endswith("90 mm²")
    dialog.pixel_size = (1e5, 1e5)
    dialog.updateStats()
    assert dialog.label_area.text().endswith("9000 cm²")

    dialog.contextMenuEvent(
        QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0))
    )

    dialog.copyToClipboard()
