import numpy as np
from pewlib.calibration import Calibration
from pewlib.config import Config
from pewlib.laser import Laser
from pewlib.srr.config import SRRConfig
from PySide6 import QtCore, QtGui
from pytestqt.qtbot import QtBot

from pewpew.graphics.imageitems import LaserImageItem
from pewpew.graphics.options import GraphicsOptions
from pewpew.lib.numpyqt import array_to_image
from pewpew.widgets import dialogs


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

    assert dialog.combo_element.currentText() == "B"
    assert dialog.points.model.array.size == 0
    assert not dialog.button_plot.isEnabled()
    assert not dialog.points.button_remove.isEnabled()

    dialog.lineedit_gradient.setText("1")
    dialog.lineedit_intercept.setText("2")
    dialog.lineedit_unit.setText("ppm")

    dialog.combo_element.setCurrentIndex(0)
    assert dialog.combo_element.currentText() == "A"
    assert dialog.points.model.array.size == 2
    assert dialog.button_plot.isEnabled()

    # Points enabled on remove / add
    dialog.points.removeCalibrationLevel()
    assert not dialog.button_plot.isEnabled()
    dialog.points.addCalibrationLevel()
    assert not dialog.button_plot.isEnabled()
    dialog.points.model.setData(dialog.points.model.index(0, 1), 1.0)
    dialog.points.model.setData(dialog.points.model.index(1, 1), 6.0)
    assert dialog.button_plot.isEnabled()
    assert np.isclose(dialog.calibrations["A"].gradient, 4.0)

    # Points weightings
    assert dialog.calibrations["A"].weighting == "Equal"
    assert np.all(dialog.calibrations["A"].weights == 1.0)

    dialog.points.combo_weighting.setCurrentText("y")
    assert dialog.calibrations["A"].weighting == "y"
    assert np.all(dialog.calibrations["A"].weights == dialog.calibrations["A"].y)

    # Restored on change
    assert dialog.calibrations["B"].gradient == 1.0
    assert dialog.calibrations["B"].intercept == 2.0
    assert dialog.calibrations["B"].unit == "ppm"

    # Just run code as can't test clipboard
    dialog.copyToClipboard()
    dialog.copyAllToClipboard()

    dialog.combo_element.setCurrentIndex(0)
    dialog.showCurve()

    dialog.apply()
    dialog.check_all.setChecked(True)
    dialog.apply()


def test_calibration_curve_dialog(qtbot: QtBot):
    dialog = dialogs.CalibrationCurveDialog(
        "A", Calibration.from_points([[0, 1], [1, 2], [2, 3], [4, 4]])
    )
    qtbot.addWidget(dialog)
    dialog.open()


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
    assert dialog.label_costes_m1.text() == "0.00"
    assert dialog.label_costes_m2.text() == "0.79"
    assert dialog.label_costes_p.text() == ""

    dialog.combo_name2.setCurrentText("a")

    assert dialog.label_r.text() == "1.00"
    assert dialog.label_icq.text() == "0.50"
    assert dialog.label_costes_m1.text() == "1.00"
    assert dialog.label_costes_m2.text() == "1.00"
    assert dialog.label_costes_p.text() == ""

    dialog.calculatePearsonsProbablity()
    assert dialog.label_costes_p.text() == "0.00"


def test_colorrange_dialog(qtbot: QtBot):
    default_range = (0.0, 1.0)
    ranges = {"A": (1.0, 2.0), "B": ("2%", 3.0)}

    dialog = dialogs.ColorRangeDialog(ranges, default_range, ["A", "B", "C"], "C")
    qtbot.addWidget(dialog)
    dialog.open()

    # Loads C as current, has default range
    assert dialog.combo_element.currentText() == "C"
    assert dialog.lineedit_min.text() == ""
    assert dialog.lineedit_max.text() == ""
    assert dialog.lineedit_min.placeholderText() == "0.0"
    assert dialog.lineedit_max.placeholderText() == "1.0"
    # Not added yet
    assert "C" not in dialog.ranges
    # Add and check is there
    dialog.lineedit_min.setText("1%")
    dialog.lineedit_max.setText("2%")
    dialog.combo_element.setCurrentText("B")  # Update C
    assert dialog.ranges["C"] == ("1%", "2%")

    assert dialog.lineedit_min.text() == "2%"
    assert dialog.lineedit_max.text() == "3.0"

    dialog.combo_element.setCurrentText("A")
    assert dialog.lineedit_min.text() == "1.0"
    assert dialog.lineedit_max.text() == "2.0"

    dialog.check_all.click()
    dialog.lineedit_min.setText("1.0")
    dialog.lineedit_max.setText("2.0")
    # dialog.combo_element.setCurrentText("C")

    dialog.apply()

    assert dialog.default_range == (1.0, 2.0)
    assert dialog.ranges == {}


def test_config_dialog(qtbot: QtBot):
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

    # Just run code as can't test clipboard
    dialog.copyToClipboard()

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


def test_info_dialog(qtbot: QtBot):
    dialog = dialogs.InformationDialog({"a": "A", "b": "B"})
    qtbot.addWidget(dialog)
    dialog.open()

    assert dialog.table.rowCount() == 3
    dialog.table.item(2, 0).setText("b")
    assert dialog.table.rowCount() == 4
    assert not dialog.isComplete()
    dialog.table.item(2, 0).setText("c")
    assert dialog.isComplete()
    dialog.table.item(2, 1).setText("C")
    assert dialog.table.rowCount() == 4

    with qtbot.waitSignal(dialog.infoChanged) as emit:
        dialog.accept()

    assert emit.args[0] == {"a": "A", "b": "B", "c": "C"}


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


def test_processing_dialog(qtbot: QtBot):
    x = np.empty((10, 10), dtype=[("x", float), ("y", float)])
    np.random.seed(86723)  # to make sure filter changes data
    x["x"] = np.random.random((10, 10))
    x["y"] = x["x"] + 10
    laser = Laser(
        data=x,
        info={
            "Name": "proctest",
            "Processing": "Filter(*,Local Median,size=3,k=1.0);Calculator(y,+ x 10)",
        },
    )
    item = LaserImageItem(laser, GraphicsOptions())
    item.redraw()

    dialog = dialogs.ProcessingDialog(["x", "y"], [item])
    qtbot.addWidget(dialog)
    dialog.open()

    assert not dialog.isComplete()
    assert dialog.list.count() == 0
    assert dialog.apply_list.count() == 0

    dialog.action_add_all_laser.trigger()
    assert dialog.apply_list.count() == 1
    assert dialog.apply_list.item(0).text() == "proctest"

    # simulate load
    dialog.loadFromString(laser.info["Processing"])
    assert dialog.list.count() == 2

    assert dialog.isComplete()

    while dialog.list.count() > 0:
        w = dialog.list.itemWidget(dialog.list.item(0))
        assert isinstance(w, dialogs.ProcessItemWidget)
        w.button_close.defaultAction().trigger()

    assert dialog.list.count() == 0
    assert not dialog.isComplete()

    proc = dialog.addCalculatorProcess()
    proc.lineedit_name.setText("z")
    proc.lineedit_expr.setText("+ x y")

    assert dialog.list.count() == 1

    proc = dialog.addFilterProcess()
    proc.combo_filter.setCurrentText("Local Mean")
    proc.combo_names.setCurrentText("y")
    proc.lineedit_fparams[0].setText("7")
    proc.lineedit_fparams[1].setText("1.5")

    assert dialog.list.count() == 2
    assert dialog.isComplete()
    dialog.accept()

    assert "z" in laser.elements
    assert np.all(laser.data["z"] == x["x"] + x["y"])
    assert np.any(laser.data["y"] != x["y"])  # changed


def test_selection_dialog(qtbot: QtBot):
    x = np.empty((10, 10), dtype=[("x", float)])
    x["x"] = np.random.random((10, 10))
    laser = Laser(data=x, info={"Name": "sel"})
    item = LaserImageItem(laser, GraphicsOptions())
    item.redraw()

    dialog = dialogs.SelectionDialog(item)
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
    assert dialog.spinbox_method.value() == 3
    assert dialog.spinbox_comparison.value() == 1

    dialog.combo_method.setCurrentText("Mean")
    dialog.check_limit_selection.setChecked(True)
    dialog.refresh()
    assert not dialog.spinbox_method.isEnabled()
    assert not dialog.spinbox_comparison.isEnabled()

    # Test correct states and masks emmited
    with qtbot.wait_signal(dialog.maskSelected) as emitted:
        dialog.apply()
    assert np.all(emitted.args[0] == (x["x"] > x["x"].mean()))
    assert emitted.args[1] == ["intersect"]

    dialog.check_limit_selection.setChecked(False)
    dialog.combo_method.setCurrentText("Manual")
    dialog.lineedit_manual.setText("0.9")
    dialog.refresh()

    with qtbot.wait_signal(dialog.maskSelected) as emitted:
        dialog.apply()
    assert np.all(emitted.args[0] == (x["x"] > 0.9))
    assert emitted.args[1] == [""]

    dialog.item.selection = emitted.args[0]

    # Test limit threshold
    dialog.combo_method.setCurrentText("Mean")
    dialog.check_limit_threshold.setChecked(True)
    item.mask_image = array_to_image((x["x"] > 0.9).astype(np.uint8))
    dialog.refresh()

    with qtbot.wait_signal(dialog.maskSelected) as emitted:
        dialog.apply()
    assert np.all(emitted.args[0] == (x["x"] > np.mean(x["x"][x["x"] > 0.9])))
    assert emitted.args[1] == [""]


def test_stats_dialog(qtbot: QtBot):
    x = np.array(np.random.random([10, 10]), dtype=[("a", float)])
    x[0, :] = np.nan
    m = np.full(x.shape, True, dtype=bool)

    dialog = dialogs.StatsDialog(x, m, {"a": "u"}, "a", (1.0, 1.0))
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
        QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0), QtCore.QPoint(0, 0)
        )
    )

    dialog.copyToClipboard()


def test_transform_dialog(qtbot: QtBot):
    t = QtGui.QTransform.fromScale(2.0, 3.0)
    t.translate(1.0, 2.0)
    p = QtCore.QPointF(2.0, 40.0)
    dialog = dialogs.TransformDialog(t, p)

    qtbot.addWidget(dialog)
    dialog.open()

    assert dialog.matrix.item(0, 0).text() == "2.0"
    assert dialog.matrix.item(1, 1).text() == "3.0"
    assert dialog.matrix.item(0, 2).text() == "6.0"
    assert dialog.matrix.item(1, 2).text() == "126.0"

    with qtbot.assert_not_emitted(dialog.transformChanged):
        dialog.accept()

    dialog.matrix.item(0, 0).setText("3.0")

    def check_transform(t: QtGui.QTransform) -> bool:
        if t.m11() != 3.0:
            return False
        if t.m31() != 0.0:
            return False
        if t.m32() != 6.0:
            return False
        return True

    with qtbot.wait_signal(
        dialog.transformChanged, timeout=500, check_params_cb=check_transform
    ):
        dialog.accept()
