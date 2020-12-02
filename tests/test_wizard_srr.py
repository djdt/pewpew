from pytestqt.qtbot import QtBot
from pathlib import Path


from pewlib.srr import SRRConfig

from pewpew.widgets.wizards.srr import (
    SRRImportWizard,
    SRRConfigPage,
    SRRPathAndOptionsPage,
)


def test_wizard_srr_import_text(qtbot: QtBot):
    path = Path(__file__).parent.joinpath("data", "io")
    wiz = SRRImportWizard(
        [path.joinpath("textimage", "srr1.csv"), path.joinpath("textimage", "srr2.csv")]
    )
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitForWindowShown(wiz)

    # Format
    page = wiz.currentPage()
    page.radio_text.setChecked(True)
    wiz.next()
    assert wiz.currentId() == wiz.page_text

    # Path and Options
    wiz.next()

    # Config
    page = wiz.currentPage()
    page.lineedit_warmup.setText("0")

    with qtbot.waitSignal(wiz.laserImported) as emit:
        wiz.accept()
        len(emit.args[0].data) == 2


# def test_wizard_srr_config(qtbot: QtBot):
#     page = SRRConfigPage(SRRConfig())
#     pass
