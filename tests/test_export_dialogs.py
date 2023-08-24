import tempfile
from pathlib import Path

from pewlib.laser import Laser
from pytestqt.qtbot import QtBot
from testing import rand_data

from pewpew.graphics.imageitems import LaserImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions
from pewpew.widgets.exportdialogs import ExportAllDialog, ExportDialog, VtiOptionsBox


def test_export_options(qtbot: QtBot):
    # png = PngOptionsBox()
    # assert not png.raw()
    # png.check_raw.setChecked(True)
    # assert png.raw()

    vti = VtiOptionsBox((10.0, 20.0, 30.0))
    assert vti.lineedits[0].text() == "10.0"
    assert vti.spacing() == (10.0, 20.0, 30.0)
    assert vti.isComplete()

    vti.lineedits[0].setText("10.0a")
    assert not vti.isComplete()


def test_export_dialog(qtbot: QtBot):
    item = LaserImageItem(
        Laser(
            rand_data("A1"), info={"Name": "laser", "File Path": "/home/user/laser.npz"}
        ),
        GraphicsOptions(),
    )
    dlg = ExportDialog(item)
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
    assert dlg.lineedit_preview.text() == "laser_<element>.png"

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
        paths = dlg.generatePaths(item)
        assert paths == [(Path(tempdir, "temp.npz"), "A1")]
        dlg.export(paths[0][0], item.laser, paths[0][1])
        assert Path(tempdir, "temp.npz").exists()
        # Test export all elements and png

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
    item = LaserImageItem(
        Laser(
            rand_data(["A", "\\B", "C>_<"]),
            info={"Name": "inv@|d", "File Path": "/invalid.npz"},
        ),
        GraphicsOptions(),
    )
    dlg = ExportDialog(item)
    dlg.lineedit_directory.setText(str(Path(".")))
    assert dlg.isComplete()
    dlg.lineedit_filename.setText("invalid.csv")
    assert dlg.isComplete()
    dlg.check_export_all.setChecked(True)

    paths = dlg.generatePaths(item)

    assert paths[0][0].name == "invalid_A.csv"
    assert paths[1][0].name == "invalid__B.csv"
    assert paths[2][0].name == "invalid_C___.csv"


def test_export_all_dialog(qtbot: QtBot):
    lasers = [
        Laser(
            rand_data("A1"),
            info={"Name": "laser1", "File Path": "/fake/directory/laser1.npz"},
        ),
        Laser(
            rand_data("B2"),
            info={"Name": "laser2", "File Path": "/fake/directory/laser2.npz"},
        ),
        Laser(
            rand_data("C3"),
            info={"Name": "laser3", "File Path": "/fake/directory/laser3.npz"},
        ),
        Laser(
            rand_data(["B2", "C3"]),
            info={"Name": "laser4", "File Path": "/fake/directory/laser4.npz"},
        ),
    ]
    options = GraphicsOptions()
    items = [LaserImageItem(laser, options) for laser in lasers]

    dlg = ExportAllDialog(items)
    dlg.open()

    assert dlg.lineedit_directory.text() == "/fake/directory"
    assert dlg.lineedit_filename.text() == "<name>.npz"
    assert dlg.lineedit_preview.text() == "<name>.npz"
    assert not dlg.isComplete()

    dlg.lineedit_prefix.setText("01")

    assert dlg.lineedit_preview.text() == "01_<name>.npz"
    assert not dlg.check_export_all.isEnabled()
    assert not dlg.check_calibrate.isEnabled()
    assert not dlg.combo_element.isEnabled()

    dlg.options.setCurrentIndex(dlg.options.indexForExt(".csv"))
    assert dlg.lineedit_preview.text() == "01_<name>.csv"
    assert dlg.check_export_all.isEnabled()
    assert dlg.check_calibrate.isEnabled()
    assert dlg.combo_element.isEnabled()

    dlg.check_export_all.click()
    assert dlg.lineedit_preview.text() == "01_<name>_<element>.csv"
    assert not dlg.combo_element.isEnabled()

    with tempfile.TemporaryDirectory() as tempdir:
        dlg.lineedit_directory.setText(tempdir)
        assert dlg.isComplete()
        dlg.accept()

        assert Path(tempdir, "01_laser1_A1.csv").exists()
        assert Path(tempdir, "01_laser2_B2.csv").exists()
        assert Path(tempdir, "01_laser3_C3.csv").exists()
        assert Path(tempdir, "01_laser4_B2.csv").exists()
        assert Path(tempdir, "01_laser4_C3.csv").exists()
