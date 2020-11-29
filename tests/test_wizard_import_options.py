from PySide2 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from pathlib import Path

from pewpew.widgets.wizards import options

path = Path(__file__).parent.joinpath("data", "io")


def test_wizard_import_options(qtbot: QtBot):
    # Agilent
    agilent = options.AgilentOptions()
    agilent.fieldArgs()
    assert not agilent.isComplete()

    agilent.updateForPath(path.joinpath("agilent", "test_ms.b"))
    assert agilent.current_path.samefile(path.joinpath("agilent", "test_ms.b"))

    assert agilent.combo_dfile_method.count() == 4
    assert agilent.combo_dfile_method.currentText() == "Batch Log XML"
    assert agilent.check_name_acq_xml.isEnabled()
    assert agilent.isComplete()

    for i in range(4):
        agilent.combo_dfile_method.setCurrentIndex(i)
        agilent.combo_dfile_method.activated.emit(i)
        assert agilent.actual_datafiles == 5
        assert agilent.expected_datafiles == 5

    agilent.updateForPath(path)
    assert agilent.combo_dfile_method.count() == 1
    assert not agilent.check_name_acq_xml.isEnabled()

    agilent.setEnabled(False)
    assert not agilent.combo_dfile_method.isEnabled()

    # Numpy
    numpy = options.NumpyOptions()
    numpy.fieldArgs()

    # PerkinElmer
    perkin = options.PerkinElmerOptions()
    perkin.fieldArgs()
    assert not perkin.isComplete()

    perkin.updateForPath(path.joinpath("perkinelmer", "perkinelmer"))
    assert perkin.isComplete()
    assert perkin.datafiles == 3

    # Text
    text = options.TextOptions()
    text.fieldArgs()
    assert text.isComplete()

    text.lineedit_name.setText("")
    assert not text.isComplete()

    # Thermo
    thermo = options.ThermoOptions()
    thermo.fieldArgs()

    thermo.updateForPath(path.joinpath("thermo", "icap_columns.csv"))
    assert thermo.combo_delimiter.currentText() == ","
    assert not thermo.radio_rows.isChecked()
    assert thermo.radio_columns.isChecked()
    assert thermo.check_use_analog.isEnabled()
    assert thermo.isComplete()

    thermo.updateForPath(path.joinpath("textimage", "csv.csv"))
    assert not thermo.check_use_analog.isEnabled()
    assert not thermo.isComplete()

    thermo.setEnabled(False)
    assert not thermo.radio_columns.isEnabled()
    assert not thermo.radio_rows.isEnabled()
    assert not thermo.combo_delimiter.isEnabled()
    assert not thermo.combo_decimal.isEnabled()


def test_wizard_import_path_widget_file(qtbot: QtBot):
    widget = options.PathSelectWidget(path, "Numpy Archive", [".npz"], "File")
    qtbot.addWidget(widget)
    assert not widget.isComplete()

    widget.addPath(path.joinpath("npz", "noreal.npz"))
    assert not widget.isComplete()

    widget.addPath(path.joinpath("npz", "test.npz"))
    assert Path(widget.lineedit_path.text()).samefile(path.joinpath("npz", "test.npz"))
    assert widget.isComplete()

    dlg = widget.selectPath()
    assert dlg.fileMode() == QtWidgets.QFileDialog.ExistingFile
    dlg.close()


def test_wizard_import_path_widget_directory(qtbot: QtBot):
    widget = options.PathSelectWidget(path, "Agilent Batch", [".b"], "Directory")
    qtbot.addWidget(widget)
    assert not widget.isComplete()

    drag_mime = QtCore.QMimeData()
    drag_mime.setUrls(
        [
            QtCore.QUrl.fromLocalFile(
                str(path.joinpath("agilent", "test_ms.b").resolve())
            )
        ]
    )
    drag_event = QtGui.QDragEnterEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.CopyAction,
        drag_mime,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    widget.dragEnterEvent(drag_event)
    assert drag_event.isAccepted()
    drop_event = QtGui.QDropEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.CopyAction,
        drag_mime,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    with qtbot.waitSignal(widget.pathChanged):
        widget.dropEvent(drop_event)
    assert drop_event.isAccepted()

    assert Path(widget.lineedit_path.text()).samefile(
        path.joinpath("agilent", "test_ms.b")
    )
    assert widget.isComplete()

    dlg = widget.selectPath()
    assert dlg.fileMode() == QtWidgets.QFileDialog.Directory
    dlg.close()


def test_wizard_import_path_widget_multiple_file(qtbot: QtBot):
    widget = options.MultiplePathSelectWidget([], "Numpy Archive", [".npz"], "File")
    qtbot.addWidget(widget)
    assert not widget.isComplete()

    widget.addPath(path.joinpath("npz", "test.npz"))
    assert Path(widget.paths[0]).samefile(path.joinpath("npz", "test.npz"))
    assert len(widget.paths) == 1
    assert widget.isComplete()

    widget.addPaths([path.joinpath("npz", "test.npz")])
    assert len(widget.paths) == 2

    widget.addPathsInDirectory(path.joinpath("npz"))
    assert len(widget.paths) == 3

    dlg = widget.selectMultiplePaths()
    assert dlg.fileMode() == QtWidgets.QFileDialog.ExistingFiles
    dlg.close()

    dlg = widget.selectAllInDirectory()
    assert dlg.fileMode() == QtWidgets.QFileDialog.Directory
    dlg.close()


def test_wizard_import_path_widget_multiple_directory(qtbot: QtBot):
    widget = options.MultiplePathSelectWidget([], "PerkinElmer XL", [""], "Directory")
    qtbot.addWidget(widget)
    assert not widget.isComplete()

    widget.addPath(path.joinpath("perkinelmer", "perkinelmer"))
    assert Path(widget.paths[0]).samefile(path.joinpath("perkinelmer", "perkinelmer"))
    assert len(widget.paths) == 1
    assert widget.isComplete()

    widget.addPathsInDirectory(path.joinpath("perkinelmer"))
    assert len(widget.paths) == 2

    # Test delete
    widget.list.selectAll()
    key_event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress, QtCore.Qt.Key_Backspace, QtCore.Qt.NoModifier
    )
    widget.keyPressEvent(key_event)
    assert len(widget.paths) == 0

    dlg = widget.selectMultiplePaths()
    assert dlg.fileMode() == QtWidgets.QFileDialog.Directory
    dlg.close()


def test_wizard_import_path_and_options_widget(qtbot: QtBot):
    widget = options.PathAndOptionsPage([Path()], "numpy", multiplepaths=False)
    qtbot.addWidget(widget)
    widget.initializePage()
    assert not widget.isComplete()
    widget.close()

    widget = options.PathAndOptionsPage(
        [path.joinpath("npz", "test.npz")], "numpy", multiplepaths=True
    )
    qtbot.addWidget(widget)
    widget.initializePage()
    assert widget.isComplete()
    widget.close()
