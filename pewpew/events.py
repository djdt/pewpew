from PySide2 import QtCore


class MousePressRedirectFilter(QtCore.QObject):
    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self.parent().mousePressEvent(event)
            return False
        return bool(super().eventFilter(obj, event))


class DragDropRedirectFilter(QtCore.QObject):
    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.DragEnter:
            self.parent().dragEnterEvent(event)
            return False
        elif event.type() == QtCore.QEvent.DragLeave:
            self.parent().dragLeaveEvent(event)
            return False
        elif event.type() == QtCore.QEvent.DragMove:
            self.parent().dragMoveEvent(event)
            return False
        elif event.type() == QtCore.QEvent.Drop:
            self.parent().dropEvent(event)
            return False
        return bool(super().eventFilter(obj, event))
