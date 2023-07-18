import tempfile
from pathlib import Path

import numpy as np
from pewlib.laser import Laser
from PySide6 import QtGui
from pytestqt.qtbot import QtBot

from pewpew.widgets.laser import LaserTabView
from pewpew.widgets.tools.overlays import OverlayTool


def test_overlay_tool(qtbot: QtBot):
    data = np.zeros((10, 10), dtype=[("r", float), ("g", float), ("b", float)])
    data["r"][:, :] = 1.0
    data["g"][:5, :] = 1.0
    data["b"][:, :5] = 1.0

    view = LaserTabView()
    qtbot.add_widget(view)
    view.show()
    widget = view.importFile(
        Laser(data, info={"Name": "test", "File Path": "/home/pewpew/real.npz"})
    )
    item = widget.laserItems()[0]
    tool = OverlayTool(item)
    view.addTab("Tool", tool)
    with qtbot.waitExposed(tool):
        tool.show()

    # Test rgb mode
    assert tool.rows.color_model == "rgb"
    assert tool.image is not None
    tool.comboAdd(1)  # r
    assert np.all(tool.image.rawData()[0, 0] == [0, 0, 255, 255])

    tool.comboAdd(2)  # g
    assert np.all(tool.image.rawData()[:5] == [0, 255, 255, 255])
    assert np.all(tool.image.rawData()[5:] == [0, 0, 255, 255])

    tool.comboAdd(3)  # b
    assert np.all(tool.image.rawData()[:5, :5] == [255, 255, 255, 255])
    assert np.all(tool.image.rawData()[5:, :5] == [255, 0, 255, 255])
    assert np.all(tool.image.rawData()[5:, 5:] == [0, 0, 255, 255])

    # Test cmyk mode
    tool.radio_cmyk.toggle()
    assert tool.rows.color_model == "cmyk"
    assert np.all(tool.image.rawData()[:5, :5] == [0, 0, 0, 255])
    assert np.all(tool.image.rawData()[5:, :5] == [0, 255, 0, 255])
    assert np.all(tool.image.rawData()[5:, 5:] == [255, 255, 0, 255])

    # Check that the rows are limited to 3
    assert tool.rows.max_rows == 3
    assert not tool.combo_add.isEnabled()
    assert tool.rows.rowCount() == 3
    with qtbot.assert_not_emitted(tool.rows.rowsChanged):
        tool.addRow("r")
    assert tool.rows.rowCount() == 3

    # Check color buttons are not enabled
    for row in tool.rows.rows:
        assert not row.button_color.isEnabled()

    # Test any mode
    tool.radio_custom.toggle()
    assert tool.rows.color_model == "any"
    assert tool.combo_add.isEnabled()
    assert tool.check_normalise.isEnabled()
    for row in tool.rows.rows:
        assert row.button_color.isEnabled()
    tool.addRow("g")
    tool.rows[3].setColor(QtGui.QColor.fromRgbF(0.0, 1.0, 1.0))
    assert tool.rows.rowCount() == 4

    # Test normalise
    assert np.amax(tool.image.rawData()[:, :, 2]) == 255
    tool.check_normalise.setChecked(True)
    tool.refresh()
    assert tool.image.rawData()[:, :, 2].max() < 255  # No red
    tool.check_normalise.setChecked(False)

    # Test export
    dlg = tool.openExportDialog()

    dlg2 = dlg.selectDirectory()
    dlg2.close()

    with tempfile.NamedTemporaryFile() as tf:
        dlg.export(Path(tf.name))
        assert Path(tf.name).exists()

    with tempfile.TemporaryDirectory() as td:
        dlg.lineedit_directory.setText(td)
        dlg.lineedit_filename.setText("test.png")
        dlg.check_individual.setChecked(True)
        dlg.accept()
        qtbot.wait(300)
        assert Path(td).joinpath("test_1.png").exists()
        assert Path(td).joinpath("test_2.png").exists()
        assert Path(td).joinpath("test_3.png").exists()

    # Test close
    with qtbot.wait_signal(tool.rows.rowsChanged):
        tool.rows.rows[-1].close()
    assert tool.rows.rowCount() == 3

    # Test hide
    tool.radio_rgb.toggle()
    with qtbot.wait_signal(tool.rows.rows[0].itemChanged):
        tool.rows.rows[0].button_hide.click()
    assert np.all(tool.image.rawData()[:, :, 2] == 0)

    dlg = tool.rows[0].selectColor()
    dlg.close()
