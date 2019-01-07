import sys

from PyQt5.QtWidgets import QApplication
from gui.mainwindow import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    sys.excepthook = window.exceptHook  # type: ignore
    window.show()

    app.exec()
