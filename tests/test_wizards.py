import os.path

from pytestqt.qtbot import QtBot


from pewpew.widgets.wizards import ImportWizard


def test_wizard_import(qtbot: QtBot):
    data_path = os.path.join(os.path.dirname(__file__), "data", "io")

    wiz = ImportWizard()
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitForWindowShown(wiz)

    # Test Agilent page
    wiz.setField("agilent", True)
    wiz.setField("agilent.path", os.path.join(data_path, "agilent.b"))
    wiz.next()
    page = wiz.currentPage()
    assert page.title() == "Agilent Batch"

    # Test button
    dlg = page.buttonPathPressed()
    dlg.close()

    # Acq method exists
    assert page.check_name_acq_xml.isEnabled()

    # Test agilent dfiles
    page.combo_dfile_method.setCurrentText("Alphabetical Order")
    assert page.dataFileCount() == (3, 3)
    page.combo_dfile_method.setCurrentText("Acquisition Method")
    assert page.dataFileCount() == (3, 3)
    page.combo_dfile_method.setCurrentText("Batch Log CSV")
    assert page.dataFileCount() == (3, 3)
    page.combo_dfile_method.setCurrentText("Batch Log XML")
    assert page.dataFileCount() == (3, 3)

    # Invalid path
    wiz.setField("agilent.path", "")
    assert not page.check_name_acq_xml.isEnabled()
    assert not page.combo_dfile_method.isEnabled()

    # Test text
    wiz.back()
    wiz.setField("text", True)
    wiz.next()

    # Test thermo
    wiz.back()
    wiz.setField("thermo", True)
    wiz.next()
