import sys

from PyQt5.QtWidgets import QApplication
from gui.qt.mainwindow import MainWindow

# TODO override close on mainwindow to cose images nicely
# TODO get some good varied colormaps

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    sys.excepthook = window.exceptHook
    window.show()

    app.exec()
