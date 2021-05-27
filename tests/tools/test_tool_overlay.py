from pathlib import Path
import numpy as np
import tempfile

from PySide2 import QtGui
from pytestqt.qtbot import QtBot

from pewlib.laser import Laser

from pewpew.widgets.laser import LaserViewSpace
from pewpew.widgets.tools.overlays import OverlayTool


def test_overlay_tool(qtbot: QtBot):
    data = np.zeros((10, 10), dtype=[("r", float), ("g", float), ("b", float)])
    data["r"][:, :] = 1.0
    data["g"][:10, :] = 1.0
    data["b"][:, :10] = 1.0

    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    view.addLaser(
        Laser(data, info={"Name": "real", "File Path": "/home/pewpew/real.npz"})
    )
    tool = OverlayTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

    # Test rgb mode
    assert tool.rows.color_model == "rgb"
    tool.comboAdd(1)  # r
    assert np.all(tool.graphics.data[0, 0] == (255 << 24) + (255 << 16))

    tool.comboAdd(2)  # g
    assert np.all(tool.graphics.data[:10] == (255 << 24) + (255 << 16) + (255 << 8))
    assert np.all(tool.graphics.data[10:] == (255 << 24) + (255 << 16))

    tool.comboAdd(3)  # g
    assert np.all(
        tool.graphics.data[:10, :10] == (255 << 24) + (255 << 16) + (255 << 8) + 255
    )
    assert np.all(tool.graphics.data[10:, :10] == (255 << 24) + (255 << 16))
    assert np.all(
        tool.graphics.data[10:, 10:] == (255 << 24) + (255 << 16) + (255 << 8)
    )
    assert np.all(tool.graphics.data[10:, 10:] == (255 << 24) + (255 << 16))

    # Test cmyk mode
    tool.radio_cmyk.toggle()
    assert tool.rows.color_model == "cmyk"
    assert np.all(tool.graphics.data[:10, :10] == (255 << 24))
    assert np.all(tool.graphics.data[10:, :10] == (255 << 24) + (255 << 8))
    assert np.all(tool.graphics.data[10:, 10:] == (255 << 25) + 255)
    assert np.all(tool.graphics.data[10:, 10:] == (255 << 24) + (255 << 8) + 255)

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
    assert np.amin(tool.graphics.data) > (255 << 24)
    tool.check_normalise.setChecked(True)
    tool.refresh()
    assert tool.graphics.data.min() == (255 << 24) + (255 << 8) + 255  # No red
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
    assert np.all(tool.graphics.data <= ((255 << 24) + (255 << 8) + 255))

    dlg = tool.rows[0].selectColor()
    dlg.close()
