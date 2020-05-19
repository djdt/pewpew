from pytestqt.qtbot import QtBot

from pew.laser import Laser

from pewpew.mainwindow import MainWindow

from testing import rand_data


def test_main_window_actions_empty(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert not window.action_export_all.isEnabled()
    assert not window.action_tool_edit.isEnabled()
    assert not window.action_tool_standards.isEnabled()
    assert not window.action_tool_overlay.isEnabled()

    dlg = window.actionOpen()
    dlg.close()
    dlg = window.actionImportAgilent()
    dlg.close()
    dlg = window.actionImportThermo()
    dlg.close()
    dlg = window.actionWizardSRR()
    dlg.close()
    dlg = window.actionConfig()
    dlg.close()
    dlg = window.actionColormapRange()
    dlg.close()
    dlg = window.actionFontsize()
    dlg.close()
    dlg = window.actionAbout()
    dlg.close()

    window.actionToggleCalibrate(False)
    window.actionToggleColorbar(False)
    window.actionToggleLabel(False)
    window.actionToggleScalebar(False)

    assert not window.viewspace.options.calibrate
    assert not window.viewspace.options.canvas.colorbar
    assert not window.viewspace.options.canvas.label
    assert not window.viewspace.options.canvas.scalebar

    window.actionGroupColormap(window.action_group_colormap.actions()[1])
    assert (
        window.viewspace.options.image.cmap
        == list(window.viewspace.options.image.COLORMAPS.values())[1]
    )
    window.actionGroupInterp(window.action_group_interp.actions()[1])
    assert (
        window.viewspace.options.image.interpolation
        == list(window.viewspace.options.image.INTERPOLATIONS.values())[1]
    )

    window.button_status_row.toggle()
    assert window.viewspace.options.units == "row"
    window.button_status_s.toggle()
    assert window.viewspace.options.units == "second"
    window.button_status_um.toggle()
    assert window.viewspace.options.units == "μm"


def test_main_window_actions_widget(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.viewspace.views[0].addLaser(Laser(rand_data("A1")))
    window.viewspace.refresh()

    assert window.action_export_all.isEnabled()
    assert window.action_tool_edit.isEnabled()
    assert window.action_tool_standards.isEnabled()
    assert window.action_tool_overlay.isEnabled()

    window.actionToggleColorbar(False)

    window.actionExportAll()
    window.actionToolEdit()
    window.actionToolStandards()
    window.actionToolOverlay()


def test_main_window_apply_dialogs(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.viewspace.views[0].addLaser(Laser(rand_data("A1")))
    window.viewspace.refresh()

    dlg = window.actionConfig()
    dlg.applyPressed.emit(dlg)
    dlg.close()
    dlg = window.actionColormapRange()
    dlg.applyPressed.emit(dlg)
    dlg.close()
    dlg = window.actionFontsize()
    dlg.intValueSelected.emit(5)
    dlg.close()
    assert window.viewspace.options.font.size == 5
