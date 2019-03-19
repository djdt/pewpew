import sys

from PyQt5.QtWidgets import QApplication
from pewpew.ui import MainWindow


if __name__ == "__main__":
    if sys.platform in ['win32', 'darwin']:
        from pewpew import icons
        icons.qInitResources()  # type: ignore

    app = QApplication(sys.argv)

    window = MainWindow()
    sys.excepthook = window.exceptHook  # type: ignore
    window.show()

    app.exec()
