import sys

from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui
from pewpew.ui import MainWindow
from pewpew.resource import app_icon
if sys.platform in ['win32', 'darwin']:
    from pewpew.resource import breath_icons
    QtGui.QIcon.setThemeName("breath")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    sys.excepthook = window.exceptHook  # type: ignore
    window.show()
    window.setWindowIcon(QtGui.QIcon(":/app.ico"))

    app.exec()
