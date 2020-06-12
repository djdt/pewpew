import sys
import logging
from PySide2 import QtCore, QtGui, QtWidgets

from pew import __version__ as __pew_version__
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
    logger.info(f"Pew {__pew_version__} loaded.")

    window.show()
    window.setWindowIcon(QtGui.QIcon(":/app.ico"))

    # Keep event loop active with timer
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    app.exec_()


if __name__ == "__main__":
    main(sys.argv[1:])
