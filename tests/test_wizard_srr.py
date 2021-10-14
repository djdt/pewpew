from pytestqt.qtbot import QtBot
from pathlib import Path


from pewlib.config import Config

from pewpew.widgets.wizards.srr import SRRImportWizard


def test_wizard_srr_config(qtbot: QtBot):
    path = Path(__file__).parent.joinpath("data", "io")
    wiz = SRRImportWizard(
        [
            path.joinpath("textimage", "csv.csv"),
            path.joinpath("textimage", "csv.csv"),
        ],
        config=Config(10.0, 40.0, 0.25),
    )
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
    assert page.getNames() == ("_Element_",)
    assert not page.isComplete()
    page.lineedit_warmup.setText("0")
    assert page.isComplete()

    page.lineedit_warmup.setText("-1")
    assert not page.isComplete()

    page.lineedit_speed.setText("0")
    assert not page.isComplete()

    dlg = page.buttonNamesPressed()
    dlg.close()

    page.updateNames({"_Element_": "Pew"})
    assert page.getNames() == ("Pew",)


def test_wizard_srr_import_agilent(qtbot: QtBot):
    path = Path(__file__).parent.joinpath("data", "io")
    wiz = SRRImportWizard(
        [
            path.joinpath("agilent", "test_ms.b"),
            path.joinpath("agilent", "test_ms.b"),
        ],
    )
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitExposed(wiz)

    # Format
    page = wiz.currentPage()
    page.radio_agilent.setChecked(True)
    wiz.next()
    assert wiz.currentId() == wiz.page_agilent

    wiz.next()

    with qtbot.waitSignal(wiz.laserImported) as emit:
        wiz.accept()
        assert len(emit.args[0].data) == 2


def test_wizard_srr_import_numpy(qtbot: QtBot):
    path = Path(__file__).parent.joinpath("data", "io")
    wiz = SRRImportWizard(
        [
            path.joinpath("npz", "test.npz"),
            path.joinpath("npz", "test.npz"),
        ],
    )
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitExposed(wiz)

    # Format
    page = wiz.currentPage()
    page.radio_numpy.setChecked(True)
    wiz.next()
    assert wiz.currentId() == wiz.page_numpy

    wiz.next()

    with qtbot.waitSignal(wiz.laserImported) as emit:
        wiz.accept()
        assert len(emit.args[0].data) == 2


def test_wizard_srr_import_text(qtbot: QtBot):
    path = Path(__file__).parent.joinpath("data", "io")
    wiz = SRRImportWizard(
        [
            path.joinpath("textimage", "srr1.csv"),
            path.joinpath("textimage", "srr2.csv"),
        ],
    )
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitExposed(wiz)

    # Format
    page = wiz.currentPage()
    page.radio_text.setChecked(True)
    wiz.next()
    assert wiz.currentId() == wiz.page_text

    wiz.next()

    with qtbot.waitSignal(wiz.laserImported) as emit:
        wiz.accept()
        assert len(emit.args[0].data) == 2


def test_wizard_srr_import_thermo(qtbot: QtBot):
    path = Path(__file__).parent.joinpath("data", "io")
    wiz = SRRImportWizard(
        [
            path.joinpath("thermo", "icap_columns.csv"),
            path.joinpath("thermo", "icap_columns.csv"),
        ],
    )
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitExposed(wiz)

    # Format
    page = wiz.currentPage()
    page.radio_thermo.setChecked(True)
    wiz.next()
    assert wiz.currentId() == wiz.page_thermo

    wiz.next()

    with qtbot.waitSignal(wiz.laserImported) as emit:
        wiz.accept()
        assert len(emit.args[0].data) == 2
