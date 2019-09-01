from pytestqt.qtbot import QtBot

from pewpew.main import MainWindow


def test_main_window_dialogs(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    dlg = window.menuOpen()
    dlg.close()
    dlg = window.menuImportAgilent()
    dlg.close()
    dlg = window.menuImportThermoiCap()
    dlg.close()
    dlg = window.menuImportKrissKross()
    dlg.close()
    dlg = window.menuExportAll()
    assert dlg is None
    dlg = window.menuColormapRange()
    dlg.close()
