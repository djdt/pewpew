import os.path
import tempfile

from pytestqt.qtbot import QtBot

from pew.laser import Laser

from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.exportdialogs import ExportDialog, ExportAllDialog

from testing import rand_data


def test_export_dialog(qtbot: QtBot):
    dlg = ExportDialog(
        Laser(rand_data("A1"), name="laser", path="/home/user/laser.npz"),
        "A1",
        (0, 100, 0, 100),
        ViewOptions(),
    )
    qtbot.addWidget(dlg)
    dlg.open()

    assert dlg.lineedit_directory.text() == "/home/user"
    assert dlg.lineedit_filename.text() == "laser.npz"
    assert dlg.lineedit_preview.text() == "laser.npz"
    assert dlg.options.currentIndex() == dlg.options.npz

    assert not dlg.check_export_all.isEnabled()
    assert not dlg.check_calibrate.isEnabled()

    dlg.lineedit_filename.setText("laser.png")
    assert dlg.options.currentIndex() == dlg.options.png

    dlg.check_export_all.setChecked(True)
    assert dlg.lineedit_preview.text() == "laser_<ISOTOPE>.png"

    dlg.lineedit_filename.setText("laser.npz")
    assert dlg.options.currentIndex() == dlg.options.npz
    assert not dlg.check_export_all.isEnabled()
    assert dlg.lineedit_preview.text() == "laser.npz"

    dir_dlg = dlg.selectDirectory()
    dir_dlg.close()

    dlg.lineedit_filename.setText("")
    assert not dlg.isComplete()

    with tempfile.TemporaryDirectory() as tempdir:
        # Test export
        dlg.lineedit_directory.setText(tempdir)
        dlg.lineedit_filename.setText("temp.npz")
        paths = dlg.generatePaths(dlg.laser)
        assert paths == [(os.path.join(tempdir, "temp.npz"), "A1")]
        dlg.export(paths, dlg.laser, dlg.viewlimits)
        assert os.path.exists(os.path.join(tempdir, "temp.npz"))
        # Test export all isotopes and png

    with tempfile.TemporaryDirectory() as tempdir:
        dlg.lineedit_directory.setText(tempdir)
        dlg.lineedit_filename.setText("temp.png")
        dlg.accept()
        assert os.path.exists(os.path.join(tempdir, "temp_A1.png"))

    dlg.close()


def test_export_all_dialog(qtbot: QtBot):
    dlg = ExportAllDialog(
        [
            Laser(rand_data("A1"), name="laser1", path="/home/user/laser1.npz"),
            Laser(rand_data("B2"), name="laser2", path="/home/user/laser2.npz"),
            Laser(rand_data("C3"), name="laser3", path="/home/user/laser3.npz"),
            Laser(rand_data(["B2", "C3"]), name="laser4", path="/home/user/laser4.npz"),
        ],
        ["A1", "B2", "C3"],
        ViewOptions(),
    )
    qtbot.addWidget(dlg)
    dlg.open()

    assert dlg.lineedit_directory.text() == "/home/user"
    assert dlg.lineedit_filename.text() == "<NAME>.npz"
    assert dlg.lineedit_preview.text() == "<NAME>.npz"

    dlg.lineedit_prefix.setText("01")

    assert dlg.lineedit_preview.text() == "01_<NAME>.npz"
    assert not dlg.check_export_all.isEnabled()
    assert not dlg.check_calibrate.isEnabled()
    assert not dlg.combo_isotopes.isEnabled()

    dlg.combo_type.setCurrentIndex(dlg.options.csv)
    assert dlg.lineedit_preview.text() == "01_<NAME>.csv"
    assert dlg.check_export_all.isEnabled()
    assert dlg.check_calibrate.isEnabled()
    assert dlg.combo_isotopes.isEnabled()

    dlg.check_export_all.setChecked(True)
    assert dlg.lineedit_preview.text() == "01_<NAME>_<ISOTOPE>.csv"
    assert not dlg.combo_isotopes.isEnabled()

    with tempfile.TemporaryDirectory() as tempdir:
        dlg.lineedit_directory.setText(tempdir)
        dlg.accept()
        assert os.path.exists(os.path.join(tempdir, "01_laser1_A1.csv"))
        assert os.path.exists(os.path.join(tempdir, "01_laser2_B2.csv"))
        assert os.path.exists(os.path.join(tempdir, "01_laser3_C3.csv"))
        assert os.path.exists(os.path.join(tempdir, "01_laser4_B2.csv"))
        assert os.path.exists(os.path.join(tempdir, "01_laser4_C3.csv"))
