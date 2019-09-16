import numpy as np
import os.path
import tempfile

from pytestqt.qtbot import QtBot

from laserlib.laser import Laser

from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.exportdialogs import ExportDialog, ExportAllDialog


def rand_laser(names: str, path: str) -> Laser:
    dtype = [(name, float) for name in names]
    return Laser.from_structured(
        np.array(np.random.random((5, 5)), dtype=dtype),
        filepath=path,
        name=os.path.splitext(os.path.basename(path))[0],
    )


def test_export_dialog(qtbot: QtBot):
    dlg = ExportDialog(
        rand_laser(["A1"], "/home/user/laser.npz"),
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
    dlg.check_export_all.clicked.emit()
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
        dlg.lineedit_directory.setText(tempdir)
        dlg.lineedit_filename.setText("temp.png")
        paths = dlg.generatePaths(dlg.laser)
        assert paths == [(os.path.join(tempdir, "temp_A1.png"), "A1")]
        dlg.export(paths, dlg.laser, dlg.viewlimits)
        assert os.path.exists(os.path.join(tempdir, "temp_A1.png"))

    dlg.close()


def test_export_all_dialog(qtbot: QtBot):
    dlg = ExportAllDialog(
        [
            rand_laser(["A1"], "/home/user/laser.npz"),
            rand_laser(["B2"], "/home/user/laser2.npz"),
            rand_laser(["C3"], "/home/user/laser3.npz"),
            rand_laser(["B2", "C3"], "/home/user/laser4.npz"),
        ],
        ["A1", "B2", "C3"],
        ViewOptions(),
    )
    qtbot.addWidget(dlg)
    dlg.open()

    assert dlg.lineedit_directory.text() == "/home/user"
    assert dlg.lineedit_filename.text() == "<NAME>.npz"
    assert dlg.lineedit_preview.text() == "<NAME>.npz"

    dlg.lineedit_prefix.setText("0123")

    assert dlg.lineedit_preview.text() == "0123_<NAME>.npz"

    assert not dlg.check_export_all.isEnabled()
    assert not dlg.check_calibrate.isEnabled()
    assert not dlg.combo_isotopes.isEnabled()

    dlg.combo_type.setCurrentIndex(dlg.options.csv)
    assert dlg.lineedit_preview.text() == "0123_<NAME>_<ISOTOPE>.csv"
    assert dlg.check_export_all.isEnabled()
    assert dlg.check_calibrate.isEnabled()
    assert dlg.combo_isotopes.isEnabled()

    dlg.check_export_all.setChecked(True)
    dlg.check_export_all.clicked.emit()
    assert not dlg.combo_isotopes.isEnabled()

    with tempfile.TemporaryDirectory() as tempdir:
        dlg.lineedit_directory.setText(tempdir)
