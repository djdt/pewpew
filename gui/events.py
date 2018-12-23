from PyQt5 import QtCore


class MousePressRedirectFilter(QtCore.QObject):
    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if isinstance(event, QtCore.QEvent.MouseButtonPress):
            self.parent().mousePressEvent(event)
            return False
        return bool(super().eventFilter(obj, event))
