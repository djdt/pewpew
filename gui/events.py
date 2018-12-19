from PyQt5 import QtCore


class MousePressRedirectFilter(QtCore.QObject):
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self.parent().mousePressEvent(event)
            return False
        return super().eventFilter(obj, event)
