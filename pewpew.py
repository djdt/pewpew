import sys

from PyQt5.QtWidgets import QApplication
from gui.mainwindow import MainWindow

VERSION = "0.3.0"

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow(VERSION)
    sys.excepthook = window.exceptHook
    window.show()

    app.exec()
