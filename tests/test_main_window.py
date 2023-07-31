from pathlib import Path

from pewlib.laser import Laser
from pytestqt.qtbot import QtBot
from testing import rand_data

from pewpew.mainwindow import MainWindow


def test_main_window_actions_empty(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert not window.action_export_all.isEnabled()

    dlg = window.actionOpen()
    dlg.close()
    dlg = window.actionWizardImport()
    dlg.close()
    # dlg = window.actionWizardSRR()
    # dlg.close()
    dlg = window.actionAbout()
    dlg.close()

    window.button_status_index.toggle()
    assert window.tabview.options.units == "index"
    window.button_status_um.toggle()
    assert window.tabview.options.units == "μm"


def test_main_window_actions_widget(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.tabview.importFile(
        Path("/home/pewpew/real.npz"),
        Laser(
            rand_data("A1"), info={"File Path": "/home/pewpew/real.npz", "Name": "test"}
        ),
    )
    window.tabview.refresh()

    assert window.action_export_all.isEnabled()

    window.actionExportAll()

    dlg = window.dialogConfig()
    dlg.applyPressed.emit(dlg)
    dlg.close()
    dlg = window.dialogColortableRange()
    dlg.applyPressed.emit(dlg)
    dlg.close()
    dlg = window.dialogFontsize()
    dlg.intValueSelected.emit(5)
    dlg.close()

    assert window.tabview.options.font.pointSize() == 5
