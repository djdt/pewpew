import os.path

from pytestqt.qtbot import QtBot


from pewpew.widgets.wizards import ImportWizard


def test_wizard_import_agilent(qtbot: QtBot):
    data_path = os.path.join(os.path.dirname(__file__), "data", "io")

    wiz = ImportWizard()
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitForWindowShown(wiz)

    # Test Agilent page
    wiz.setField("agilent", True)
    wiz.next()
    page = wiz.currentPage()
    assert page.title() == "Agilent Batch Import"

    # Empty path
    assert not page.check_name_acq_xml.isEnabled()
    assert not page.combo_dfile_method.isEnabled()
    wiz.setField("agilent.path", os.path.join(data_path, "agilent.b"))

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

    # Config page
    wiz.next()
    page = wiz.currentPage()

    assert page.label_isotopes.text() == "A1, B2"
    assert float(wiz.field("scantime")) == 0.1

    with qtbot.wait_signal(wiz.laserImported):
        wiz.accept()

def test_wizard_import_text(qtbot: QtBot):
    data_path = os.path.join(os.path.dirname(__file__), "data", "io")

    wiz = ImportWizard()
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitForWindowShown(wiz)

    # Test text page
    wiz.setField("text", True)
    wiz.next()
    page = wiz.currentPage()
    assert page.title() == "Text Image Import"

    wiz.setField("text.path", os.path.join(data_path, "txt.txt"))

    # Test button
    dlg = page.buttonPathPressed()
    dlg.close()

    assert page.isComplete()
    wiz.setField("text.name", "")
    assert not page.isComplete()
    wiz.setField("text.name", "A1")

    wiz.next()
    page = wiz.currentPage()

    assert page.label_isotopes.text() == "A1"

    # Test the name dialog works
    dlg = page.buttonNamesPressed()
    dlg.namesSelected.emit(["A1"], ["A2"])
    dlg.close()

    assert page.label_isotopes.text() == "A2"

    with qtbot.wait_signal(wiz.laserImported):
        wiz.accept()


def test_wizard_import_thermo(qtbot: QtBot):
    data_path = os.path.join(os.path.dirname(__file__), "data", "io")

    wiz = ImportWizard()
    qtbot.addWidget(wiz)
    wiz.show()
    qtbot.waitForWindowShown(wiz)

    # Test text page
    wiz.setField("thermo", True)
    wiz.next()
    page = wiz.currentPage()
    assert page.title() == "Thermo iCap Data Import"

    wiz.setField("thermo.path", os.path.join(data_path, "icap_rows.csv"))

    # Test button
    dlg = page.buttonPathPressed()
    dlg.close()

    assert not wiz.field("thermo.sampleColumns")
    assert wiz.field("thermo.sampleRows")
    assert not wiz.field("thermo.useAnalog")

    wiz.setField("thermo.path", os.path.join(data_path, "icap_columns.csv"))

    assert wiz.field("thermo.sampleColumns")
    assert not wiz.field("thermo.sampleRows")
    assert not wiz.field("thermo.useAnalog")

    wiz.next()
    page = wiz.currentPage()

    assert page.label_isotopes.text() == "1A, 2B"
    assert float(wiz.field("scantime")) == 0.1

    with qtbot.wait_signal(wiz.laserImported):
        wiz.accept()
