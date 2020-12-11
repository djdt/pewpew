import argparse
import logging
import multiprocessing
from pathlib import Path
import sys

from PySide2 import QtCore, QtGui, QtWidgets

import pewlib
from pewpew import __version__

from pewpew.mainwindow import MainWindow
from pewpew.resources import app_icon  # noqa: F401
from pewpew.resources import icons  # noqa: F401

from typing import List


logger = logging.getLogger()


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pew²",
        description="GUI for visualisation and manipulation of LA-ICP-MS data.",
    )

    parser.add_argument("--open", "-i", type=Path, nargs="+", help="Open file(s) on startup.")
    parser.add_argument(
        "qtargs", nargs=argparse.REMAINDER, help="Arguments to pass to Qt."
    )
    args = parser.parse_args(argv)

    if args.open is not None:
        for path in args.open:
            if not path.exists:
                raise parser.error(f"[--open, -i]: File '{path}' not found.")

    return args


def main(argv: List[str] = None) -> int:
    multiprocessing.freeze_support()
    args = parse_args(argv)

    app = QtWidgets.QApplication(args.qtargs)
    app.setApplicationName("pew²")
    app.setApplicationVersion(__version__)

    window = MainWindow()
    sys.excepthook = window.exceptHook
    logger.addHandler(window.log.handler)
    logger.info(f"Pew² {__version__} started.")
    logger.info(f"Using Pewlib {pewlib.__version__}.")

    window.show()
    window.setWindowIcon(QtGui.QIcon(":/app.ico"))

    # Arguments
    if args.open is not None:
        window.viewspace.activeView().openDocument(args.open)

    # Keep event loop active with timer
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    return app.exec_()


if __name__ == "__main__":
    main(sys.argv[1:])
