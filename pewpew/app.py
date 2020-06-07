import sys
import logging
from PySide2 import QtCore, QtGui, QtWidgets

from pewpew import __version__
from pewpew.mainwindow import MainWindow
from pewpew.resources import app_icon  # noqa: F401
from pewpew.resources import icons  # noqa: F401

from typing import List


logger = logging.getLogger()


def main(args: List[str] = None) -> None:
    app = QtWidgets.QApplication(args or [])

    window = MainWindow()
    sys.excepthook = window.exceptHook
    logger.addHandler(window.log.handler)
    logger.info(f"Pewpew {__version__} started.")

    window.show()
    window.setWindowIcon(QtGui.QIcon(":/app.ico"))

    # Keep event loop active with timer
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    app.exec_()
