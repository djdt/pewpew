from pathlib import Path

import numpy as np
from pewlib.calibration import Calibration
from pewlib.config import Config
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot
from testing import rand_data

from pewpew.widgets.laser import LaserTabView, LaserTabWidget


def test_laser_tab_view(qtbot: QtBot):
    view = LaserTabView()
    qtbot.addWidget(view)
    view.show()

    assert view.activeWidget() is None

    view.addTab("tab1", LaserTabWidget(view.options, view))
    view.addTab("tab2", LaserTabWidget(view.options, view))

    view.stack.widget(0).addLaser(Laser(rand_data(["A1", "B2"]), info={"Name": "tab"}))
    view.stack.widget(0).addLaser(Laser(rand_data(["A1", "C3"]), info={"Name": "tab"}))
    view.stack.widget(1).addLaser(Laser(rand_data(["B2", "D4"]), info={"Name": "tab"}))

    assert sorted(view.uniqueElements()) == ["A1", "B2", "C3", "D4"]
    assert view.activeWidget().laser_controls.elements.currentText() == "A1"

    # Apply config
    view.applyConfig(Config(10, 10, 10))
    for widget in view.widgets():
        for item in widget.laserItems():
            assert item.laser.config.spotsize == 10
            assert item.laser.config.speed == 10
            assert item.laser.config.scantime == 10
    # Try to apply calibraiton
    view.applyCalibration({"A1": Calibration(1.0, 1.0), "B2": Calibration(2.0, 2.0)})
    qtbot.waitExposed(view)
    for widget in view.widgets():
        for item in widget.laserItems():
            if "A1" in item.laser.elements:
                assert item.laser.calibration["A1"].intercept == 1.0
                assert item.laser.calibration["A1"].gradient == 1.0
            if "B2" in item.laser.elements:
                assert item.laser.calibration["B2"].intercept == 2.0
                assert item.laser.calibration["B2"].gradient == 2.0
            if "C3" in item.laser.elements:
                assert item.laser.calibration["C3"].intercept == 0.0
                assert item.laser.calibration["C3"].gradient == 1.0

    # Check element changed if avilable
    assert view.activeWidget().laser_controls.elements.currentText() == "A1"
    assert view.stack.widget(1).laser_controls.elements.currentText() == "B2"
    view.setElement("D4")
    assert view.activeWidget().laser_controls.elements.currentText() == "A1"
    assert view.stack.widget(1).laser_controls.elements.currentText() == "D4"

    # Test global color config
    for item in view.activeWidget().laserItems():
        item.element == "A1"

    view.activeWidget().laser_controls.elements.setCurrentText("B2")
    for item in view.activeWidget().laserItems():
        item.element == "B2"

    view.activeWidget().laser_controls.element_lock.toggle()
    view.activeWidget().laser_controls.elements.setCurrentText("A1")

    for item, element in zip(view.activeWidget().laserItems(), ["A1", "B2"]):
        item.element == element

    # Close all
    for widget in view.widgets():
        widget.close()


def test_laser_tab_widget(qtbot: QtBot):
    x = rand_data(["A1", "B2"])
    y = x["A1"].copy()

    view = LaserTabView()
    qtbot.addWidget(view)
    view.show()

    widget = LaserTabWidget(view.options, view)
    view.addTab("tab1", widget)
    item = view.activeWidget().addLaser(Laser(x, info={"Name": "tab"}))

    # item.applyConfig(Config(1.0, 1.0, 1.0))
    # assert item.laser.config.spotsize == 1.0
    # item.applyCalibration({"B2": Calibration(2.0, 2.0)})
    # assert item.laser.calibration["B2"].intercept == 2.0

    # item.renameElements({"A1": "A1", "B2": "2B"})
    # assert np.all(view.uniqueElements() == ["2B", "A1"])

    # item.transform(flip="horizontal")
    # assert np.all(item.laser.get("A1") == np.flip(y, axis=1))
    # item.transform(flip="horizontal")
    # item.transform(flip="vertical")
    # assert np.all(item.laser.get("A1") == np.flip(y, axis=0))
    # item.transform(flip="vertical")
    # assert np.all(item.laser.get("A1") == y)
    # item.transform(rotate="right")
    # assert np.all(item.laser.get("A1") == np.rot90(y, k=1, axes=(1, 0)))
    # item.transform(rotate="left")
    # assert np.all(item.laser.get("A1") == y)

    view.contextMenuEvent(
        QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0), QtCore.QPoint(0, 0)
        )
    )

    # Drop event
    drag_mime = QtCore.QMimeData()
    path = Path(__file__).parent.joinpath("data", "io", "npz", "test.npz")
    drag_mime.setUrls([QtCore.QUrl.fromLocalFile(str(path.resolve()))])
    drag_event = QtGui.QDragEnterEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.CopyAction,
        drag_mime,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    view.dragEnterEvent(drag_event)
    assert drag_event.isAccepted()
    drop_event = QtGui.QDropEvent(
        QtCore.QPoint(0, 0),
        QtCore.Qt.CopyAction,
        drag_mime,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    with qtbot.waitSignal(widget.numLaserItemsChanged):
        view.dropEvent(drop_event)
    assert drop_event.isAccepted()
    assert len(widget.laserItems()) == 2


def test_laser_widget_cursor(qtbot: QtBot):
    main = QtWidgets.QMainWindow()
    qtbot.addWidget(main)
    main.statusBar()  # Create bar
    view = LaserTabView()
    main.setCentralWidget(view)

    view.importFile(
        Path("/home/pewpew/fake.npz"), Laser(rand_data(["a"]), info={"Name": "test"})
    )
    widget = view.activeWidget()

    # Cursor
    widget.updateCursorStatus(QtCore.QPointF(2.0, 2.0), QtCore.QPoint(0, 0), 1.0)
    assert main.statusBar().currentMessage() == "2,2 [1]"

    widget.updateCursorStatus(QtCore.QPointF(1.0, 3.0), QtCore.QPoint(0, 0), np.nan)
    assert main.statusBar().currentMessage() == "1,3 [nan]"
