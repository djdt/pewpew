from PySide2 import QtCore, QtGui, QtWidgets

# from pewpew.events import DragDropRedirectFilter
from pewpew.events import MousePressRedirectFilter

from pytestqt.qtbot import QtBot


class EventTestWidget(QtWidgets.QWidget):
    eventSuccess = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.eventSuccess.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        self.eventSuccess.emit()
        super().dragEnterEvent(event)

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent):
        self.eventSuccess.emit()
        super().dragLeaveEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        self.eventSuccess.emit()
        super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent):
        self.eventSuccess.emit()
        super().dropEvent(event)


def test_mouse_press_redirect_filter(qtbot: QtBot):
    a = EventTestWidget()
    b = EventTestWidget()
    qtbot.addWidget(a)
    qtbot.addWidget(b)

    b.installEventFilter(MousePressRedirectFilter(a))

    with qtbot.waitSignal(a.eventSuccess):
        qtbot.mousePress(b, QtCore.Qt.LeftButton)

    with qtbot.waitSignal(b.eventSuccess):
        qtbot.mousePress(b, QtCore.Qt.LeftButton)

    with qtbot.assertNotEmitted(a.eventSuccess):
        qtbot.keyPress(b, QtCore.Qt.Key_Enter)


def test_drag_drop_redirect_filter(qtbot: QtBot):
    # Unsure how to test this one
    pass
