from pathlib import Path
import tempfile

from pytestqt.qtbot import QtBot

from pewlib.laser import Laser

from pewpew.widgets.exportdialogs import (
    ExportDialog,
    ExportAllDialog,
    PngOptionsBox,
    VtiOptionsBox,
)
from pewpew.widgets.laser import LaserViewSpace

from testing import rand_data


def test_export_options(qtbot: QtBot):
    png = PngOptionsBox()
    assert not png.raw()
    png.check_raw.setChecked(True)
    assert png.raw()

    vti = VtiOptionsBox((10.0, 20.0, 30.0))
    assert vti.lineedits[0].text() == "10.0"
    assert vti.spacing() == (10.0, 20.0, 30.0)
    assert vti.isComplete()

    vti.lineedits[0].setText("10.0a")
    assert not vti.isComplete()


def test_export_dialog(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    view = viewspace.activeView()

    widget = view.addLaser(
        Laser(rand_data("A1"), name="laser", path=Path("/home/user/laser.npz"))
    )
    dlg = ExportDialog(widget)
    dlg.open()

    assert dlg.lineedit_directory.text() == "/home/user"
    assert dlg.lineedit_filename.text() == "laser.npz"
    assert dlg.lineedit_preview.text() == "laser.npz"
    assert dlg.options.currentExt() == ".npz"

    assert not dlg.check_export_all.isEnabled()
    assert not dlg.check_calibrate.isEnabled()

    dlg.lineedit_filename.setText("laser.png")
    assert dlg.lineedit_preview.text() == "laser.png"
    assert dlg.options.currentExt() == ".png"

    dlg.lineedit_filename.setText("laser")
    assert dlg.lineedit_preview.text() == "laser.png"
    assert dlg.options.currentExt() == ".png"

    dlg.check_export_all.click()
    assert dlg.lineedit_preview.text() == "laser_<isotope>.png"

    dlg.lineedit_filename.setText("laser.npz")
    assert dlg.options.currentExt() == ".npz"
    assert not dlg.check_export_all.isEnabled()
    assert dlg.lineedit_preview.text() == "laser.npz"

    dlg.lineedit_filename.setText("laser.abc")
    assert not dlg.isComplete()
    dlg.lineedit_filename.setText("laser.npz")

    dir_dlg = dlg.selectDirectory()
    dir_dlg.close()

    dlg.lineedit_filename.setText("/fake/directory")
    assert not dlg.isComplete()

    with tempfile.TemporaryDirectory() as tempdir:
        # Test export
        dlg.lineedit_directory.setText(tempdir)
        dlg.lineedit_filename.setText("temp.npz")
        paths = dlg.generatePaths(dlg.widget)
        assert paths == [(Path(tempdir, "temp.npz"), "A1", None)]
        dlg.export(paths[0][0], paths[0][1], None, dlg.widget)
        assert Path(tempdir, "temp.npz").exists()
        # Test export all isotopes and png

    with tempfile.TemporaryDirectory() as tempdir:
        dlg.lineedit_directory.setText(tempdir)
        dlg.lineedit_filename.setText("temp.png")
        dlg.accept()
        assert Path(tempdir, "temp_A1.png").exists()

    with tempfile.TemporaryDirectory() as tempdir:
        dlg.lineedit_directory.setText(tempdir)
        dlg.lineedit_filename.setText("temp.vti")
        dlg.accept()
        assert Path(tempdir, "temp.vti").exists()

    dlg.close()


def test_export_dialog_names(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    view = viewspace.activeView()

    widget = view.addLaser(
        Laser(
            rand_data(["A", "\\B", "C>_<"]),
            name="inv@|d",
            path=Path("/"),
        )
    )
    dlg = ExportDialog(widget)
    dlg.lineedit_directory.setText(str(Path(".")))
    assert dlg.isComplete()
    dlg.lineedit_filename.setText("invalid.csv")
    assert dlg.isComplete()
    dlg.check_export_all.setChecked(True)

    paths = dlg.generatePaths(widget.laser)

    assert paths[0][0].name == "invalid_A.csv"
    assert paths[1][0].name == "invalid__B.csv"
    assert paths[2][0].name == "invalid_C___.csv"


def test_export_all_dialog(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    view = viewspace.activeView()

    lasers = [
        Laser(rand_data("A1"), name="laser1", path=Path("/fake/directory/laser1.npz")),
        Laser(rand_data("B2"), name="laser2", path=Path("/fake/directory/laser2.npz")),
        Laser(rand_data("C3"), name="laser3", path=Path("/fake/directory/laser3.npz")),
        Laser(
            rand_data(["B2", "C3"]),
            name="laser4",
            path=Path("/fake/directory/laser4.npz"),
        ),
    ]
    widgets = [view.addLaser(laser) for laser in lasers]

    dlg = ExportAllDialog(widgets)
    dlg.open()

    assert dlg.lineedit_directory.text() == "/fake/directory"
    assert dlg.lineedit_filename.text() == "<name>.npz"
    assert dlg.lineedit_preview.text() == "<name>.npz"
    assert not dlg.isComplete()

    dlg.lineedit_prefix.setText("01")

    assert dlg.lineedit_preview.text() == "01_<name>.npz"
    assert not dlg.check_export_all.isEnabled()
    assert not dlg.check_calibrate.isEnabled()
    assert not dlg.combo_isotope.isEnabled()

    dlg.options.setCurrentIndex(dlg.options.indexForExt(".csv"))
    assert dlg.lineedit_preview.text() == "01_<name>.csv"
    assert dlg.check_export_all.isEnabled()
    assert dlg.check_calibrate.isEnabled()
    assert dlg.combo_isotope.isEnabled()

    dlg.check_export_all.click()
    assert dlg.lineedit_preview.text() == "01_<name>_<isotope>.csv"
    assert not dlg.combo_isotope.isEnabled()

    with tempfile.TemporaryDirectory() as tempdir:
        dlg.lineedit_directory.setText(tempdir)
        assert dlg.isComplete()
        dlg.accept()

        assert Path(tempdir, "01_laser1_A1.csv").exists()
        assert Path(tempdir, "01_laser2_B2.csv").exists()
        assert Path(tempdir, "01_laser3_C3.csv").exists()
        assert Path(tempdir, "01_laser4_B2.csv").exists()
        assert Path(tempdir, "01_laser4_C3.csv").exists()
