from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.events import DragDropRedirectFilter, MousePressRedirectFilter

from pytestqt.qtbot import QtBot


class EventTestWidget(QtWidgets.QWidget):
    eventSuccess = QtCore.Signal()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.eventSuccess.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        self.eventSuccess.emit()
        super().dragEnterEvent(event)

    def dragLeaveEvent(self, event):
        self.eventSuccess.emit()
        super().dragLeaveEvent(event)

    def dragMoveEvent(self, event):
        self.eventSuccess.emit()
        super().dragMoveEvent(event)

    def dropEvent(self, event):
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


def test_drag_drop_redirect_filter(qtbot: QtBot):
    a = EventTestWidget()
    b = EventTestWidget()
    qtbot.addWidget(a)
    qtbot.addWidget(b)

    b.installEventFilter(DragDropRedirectFilter(a))

    # Not sure how to test this properly
