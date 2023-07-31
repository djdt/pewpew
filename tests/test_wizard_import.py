from pytestqt.qtbot import QtBot

from pathlib import Path

from pewpew.widgets.wizards import ImportWizard

path = Path(__file__).parent.joinpath("data", "io")


def test_wizard_import_config(qtbot: QtBot):
    wiz = ImportWizard(path.joinpath("agilent", "test_ms.b"))
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitExposed(wiz)

    wiz.next()
    wiz.next()

    page = wiz.currentPage()
    assert page.isComplete()
    assert page.getNames() == ("P31", "Eu153", "W182")
    assert page.lineedit_aspect.text() == "2.00"

    page.lineedit_scantime.setText("0")
    assert not page.isComplete()
    assert page.lineedit_aspect.text() == "0.00"

    page.lineedit_speed.setText("0")
    assert not page.isComplete()

    page.lineedit_spotsize.setText("0")
    assert not page.isComplete()
    assert page.lineedit_aspect.text() == ""

    dlg = page.buttonNamesPressed()
    dlg.close()

    page.updateNames({"P31": "31P", "Eu153": "Eu153"})
    assert page.getNames() == ("31P", "Eu153")


def test_wizard_import_agilent(qtbot: QtBot):
    wiz = ImportWizard(path.joinpath("agilent", "test_ms.b"))
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitExposed(wiz)

    # Format
    page = wiz.currentPage()
    page.radio_agilent.setChecked(True)
    wiz.next()
    assert wiz.currentId() == wiz.page_agilent

    # Path and Options
    wiz.next()

    # Config
    page = wiz.currentPage()
    assert page.lineedit_scantime.text() == "0.5"

    with qtbot.waitSignal(wiz.laserImported) as emit:
        wiz.accept()
        assert emit.args[1].shape == (5, 5)


def test_wizard_import_perkinelemer(qtbot: QtBot):
    wiz = ImportWizard(path.joinpath("perkinelmer", "perkinelmer"))
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitExposed(wiz)

    # Format
    page = wiz.currentPage()
    page.radio_perkinelmer.setChecked(True)
    wiz.next()
    assert wiz.currentId() == wiz.page_perkinelmer

    # Path and Options
    wiz.next()

    # Config
    page = wiz.currentPage()
    assert page.lineedit_speed.text() == "100"
    assert page.lineedit_scantime.text() == "0.2"
    assert page.lineedit_spotsize.text() == "300"

    with qtbot.waitSignal(wiz.laserImported) as emit:
        wiz.accept()
        assert emit.args[1].shape == (3, 3)


def test_wizard_import_text(qtbot: QtBot):
    wiz = ImportWizard(path.joinpath("textimage", "csv.csv"))
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitExposed(wiz)

    # Format
    page = wiz.currentPage()
    page.radio_text.setChecked(True)
    wiz.next()
    assert wiz.currentId() == wiz.page_text

    # Path and Options
    wiz.next()

    # Config
    page = wiz.currentPage()

    with qtbot.waitSignal(wiz.laserImported) as emit:
        wiz.accept()
        assert emit.args[1].shape == (5, 5)


def test_wizard_import_thermo(qtbot: QtBot):
    wiz = ImportWizard(path.joinpath("thermo", "icap_columns.csv"))
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitExposed(wiz)

    # Format
    page = wiz.currentPage()
    page.radio_thermo.setChecked(True)
    wiz.next()
    assert wiz.currentId() == wiz.page_thermo

    # Path and Options
    wiz.next()

    # Config
    page = wiz.currentPage()
    assert page.lineedit_scantime.text() == "1.0049"

    with qtbot.waitSignal(wiz.laserImported) as emit:
        wiz.accept()
        assert emit.args[1].shape == (5, 5)
