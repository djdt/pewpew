from pytestqt.qtbot import QtBot

from pewlib.laser import Laser

from pewpew.mainwindow import MainWindow

from testing import rand_data


def test_main_window_actions_empty(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert not window.action_export_all.isEnabled()
    assert not window.action_tool_calculator.isEnabled()
    assert not window.action_tool_drift.isEnabled()
    assert not window.action_tool_filter.isEnabled()
    assert not window.action_tool_standards.isEnabled()
    assert not window.action_tool_overlay.isEnabled()

    dlg = window.actionOpen()
    dlg.close()
    dlg = window.actionWizardImport()
    dlg.close()
    dlg = window.actionWizardSRR()
    dlg.close()
    dlg = window.actionAbout()
    dlg.close()

    window.button_status_index.toggle()
    assert window.viewspace.options.units == "index"
    window.button_status_um.toggle()
    assert window.viewspace.options.units == "μm"


def test_main_window_actions_widget(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.viewspace.views[0].addLaser(
        Laser(rand_data("A1"), info={"File Path": "/home/pewpew/real.npz"})
    )
    window.viewspace.refresh()

    assert window.action_export_all.isEnabled()
    assert window.action_tool_calculator.isEnabled()
    assert window.action_tool_drift.isEnabled()
    assert window.action_tool_filter.isEnabled()
    assert window.action_tool_standards.isEnabled()
    assert window.action_tool_overlay.isEnabled()

    window.actionTransformFlipHorz()
    window.actionTransformFlipVert()
    window.actionTransformRotateLeft()
    window.actionTransformRotateRight()

    window.actionExportAll()
    window.actionToolCalculator()
    window.actionToolDrift()
    window.actionToolFilter()
    window.actionToolStandards()
    window.actionToolOverlay()

# def test_laser_view_space_apply_dialogs(qtbot: QtBot):
#     viewspace = LaserViewSpace()
#     qtbot.addWidget(viewspace)
#     viewspace.show()

#     qtbot.addWidget(viewspace)
#     viewspace.views[0].addLaser(Laser(rand_data("A1")))
#     viewspace.refresh()

#     dlg = viewspace.configDialog()
#     dlg.applyPressed.emit(dlg)
#     dlg.close()
#     dlg = viewspace.colortableRangeDialog()
#     dlg.applyPressed.emit(dlg)
#     dlg.close()
#     dlg = viewspace.fontsizeDialog()
#     dlg.intValueSelected.emit(5)
#     dlg.close()

#     assert viewspace.options.font.pointSize() == 5
