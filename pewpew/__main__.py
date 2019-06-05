import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from pewpew.ui import MainWindow
from pewpew.resource import app_icon  # noqa: F401

if sys.platform in ['win32', 'darwin']:
    from pewpew.resource import breath_icons  # noqa: F401
    QIcon.setThemeName("breath")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    sys.excepthook = window.exceptHook  # type: ignore
    window.show()
    window.setWindowIcon(QIcon(":/app.ico"))

    # Keep event loop active with timer
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    app.exec()
