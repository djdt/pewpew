import sys

from PyQt5.QtWidgets import QApplication
from pewpew.ui import MainWindow
if sys.platform in ['win32', 'darwin']:
    from pewpew import resource

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    sys.excepthook = window.exceptHook  # type: ignore
    window.show()

    app.exec()
