import numpy as np

from pytestqt.qtbot import QtBot

from laserlib.laser import Laser

from pewpew.main import MainWindow
from pewpew.widgets.laser import LaserWidget


def test_main_window_actions_empty(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert not window.action_copy_image.isEnabled()
    assert not window.action_config.isEnabled()
    assert not window.action_calibration.isEnabled()
    assert not window.action_statistics.isEnabled()
    assert not window.action_save.isEnabled()
    assert not window.action_export.isEnabled()
    assert not window.action_export_all.isEnabled()
    assert not window.action_calculations_tool.isEnabled()
    assert not window.action_standards_tool.isEnabled()

    dlg = window.actionOpen()
    dlg.close()
    dlg = window.actionImportAgilent()
    dlg.close()
    dlg = window.actionImportThermo()
    dlg.close()
    dlg = window.actionImportSRR()
    dlg.close()
    dlg = window.actionConfigDefault()
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
    window.viewspace.views[0].addTab(
        "A1",
        LaserWidget(
            Laser.from_structured(
                np.array(np.random.random((10, 10)), dtype=[("A1", float)])
            ),
            window.viewspace.options,
        ),
    )
    window.viewspace.refresh()

    assert window.action_copy_image.isEnabled()
    assert window.action_config.isEnabled()
    assert window.action_calibration.isEnabled()
    assert window.action_statistics.isEnabled()
    assert window.action_save.isEnabled()
    assert window.action_export.isEnabled()
    assert window.action_export_all.isEnabled()
    assert window.action_calculations_tool.isEnabled()
    assert window.action_standards_tool.isEnabled()

    window.actionToggleColorbar(False)

    dlg = window.actionConfig()
    dlg.close()
    dlg = window.actionCalibration()
    dlg.close()
    dlg = window.actionStatistics()
    dlg.close()
    dlg = window.actionSave()
    dlg.close()
    dlg = window.actionExport()
    dlg.close()
    dlg = window.actionExportAll()
    dlg.close()
    dlg = window.actionStandardsTool()
    dlg.close()
    dlg = window.actionCalculationsTool()
    dlg.close()

    window.actionCopyImage()


def test_main_window_apply_dialogs(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.viewspace.views[0].addTab(
        "A1",
        LaserWidget(
            Laser.from_structured(
                np.array(np.random.random((10, 10)), dtype=[("A1", float)])
            ),
            window.viewspace.options,
        ),
    )
    window.viewspace.refresh()

    dlg = window.actionConfig()
    dlg.applyPressed.emit(dlg)
    dlg.close()
    dlg = window.actionCalibration()
    dlg.applyPressed.emit(dlg)
    dlg.close()
    dlg = window.actionColormapRange()
    dlg.applyPressed.emit(dlg)
    dlg.close()
    dlg = window.actionFontsize()
    dlg.intValueSelected.emit(5)
    dlg.close()
    assert window.viewspace.options.font.size == 5
