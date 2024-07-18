from pathlib import Path

from pytestqt.qtbot import QtBot

from pewpew.widgets.wizards import LaserLogImportWizard

path = Path(__file__).parent.joinpath("data", "io")


def test_laserlog_import_wizard(qtbot: QtBot):
    wiz = LaserLogImportWizard(
        path.joinpath("nwi_laser", "LaserLog_by_line.csv"),
        [path.joinpath("nwi_laser", "nwi_laser_by_line.b")],
    )
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitExposed(wiz)

    page = wiz.currentPage()
    assert page.path._path.endswith("LaserLog_by_line.csv")

    wiz.next()
    page = wiz.currentPage()
    assert page.radio_agilent.isChecked()

    # Agilent import page
    wiz.next()
    page = wiz.currentPage()
    assert len(page.path._paths)
    assert page.path._paths[0].endswith("nwi_laser_by_line.b")

    # Grouping page
    wiz.next()
    page = wiz.currentPage()
    assert page.group_tree.topLevelItemCount() == 3 + 1
    assert page.group_tree.topLevelItem(0).childCount() == 1
    assert page.group_tree.topLevelItem(1).childCount() == 0
    assert page.group_tree.topLevelItem(2).childCount() == 0

    page.checkbox_split.click()
    assert page.group_tree.topLevelItem(0).childCount() == 1
    assert page.group_tree.topLevelItem(1).childCount() == 1
    assert page.group_tree.topLevelItem(2).childCount() == 1

    # Laser view page
    wiz.next()
    page = wiz.currentPage()
    assert page.spinbox_delay.specialValueText() == "Automatic (0.0000)"

    item_positions = [(1000.0, 1000.0), (1000.0, 1100.0), (1000.0, 1200.0)]
    for item, pos in zip(page.getLaserItems()[::-1], item_positions):
        assert item.pos().x() == pos[0]
        assert item.pos().y() == pos[1]

    page.checkbox_collapse.click()
    item_positions = [(1000.0, 1000.0), (1000.0, 1010.0), (1000.0, 1020.0)]
    for item, pos in zip(page.getLaserItems()[::-1], item_positions):
        assert item.pos().x() == pos[0]
        assert item.pos().y() == pos[1]

    with qtbot.wait_signals([wiz.laserImported, wiz.laserImported, wiz.laserImported]):
        wiz.accept()
