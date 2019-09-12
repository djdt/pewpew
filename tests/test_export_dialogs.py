import numpy as np

from pytestqt.qtbot import QtBot

from laserlib.laser import Laser

from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.exportdialogs import ExportDialog, ExportAllDialog


def test_export_dialog(qtbot: QtBot):
    dlg = ExportDialog(
        Laser.from_structured(
            np.array(np.random.random((5, 5)), dtype=[("A1", float)]),
            filepath="/home/user/laser.npz"
        ),
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

    dlg.close()


# def test_export_all_dialog(qtbot: QtBot):
#     dlg = ExportAllDialog()
