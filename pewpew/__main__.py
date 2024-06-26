import argparse
import logging
import multiprocessing
import sys
from importlib.metadata import version
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from pewpew import resources  # noqa: F401
from pewpew.mainwindow import MainWindow

logger = logging.getLogger()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pew²",
        description="GUI for visualisation and manipulation of LA-ICP-MS data.",
    )

    parser.add_argument(
        "--open", "-i", type=Path, nargs="+", help="Open file(s) on startup."
    )
    parser.add_argument(
        "--nohook", action="store_true", help="Don't install the execption hook."
    )
    parser.add_argument(
        "qtargs", nargs=argparse.REMAINDER, help="Arguments to pass to Qt."
    )
    args = parser.parse_args(argv[1:])

    if args.open is not None:
        for path in args.open:
            if not path.exists:
                raise parser.error(f"[--open, -i]: File '{path}' not found.")

    return args


def main() -> int:
    args = parse_args(sys.argv)

    app = QtWidgets.QApplication(args.qtargs)
    app.setApplicationName("pewpew")
    app.setOrganizationName("pewpew")
    app.setApplicationVersion(version("pewpew"))
    app.setWindowIcon(QtGui.QIcon(":/app.ico"))

    window = MainWindow()
    if not args.nohook:
        sys.excepthook = window.exceptHook
    logger.addHandler(window.log.handler)
    logger.info(f"Pew² {app.applicationVersion()} started.")
    logger.info(f"Using Pewlib {version('pewlib')}.")

    window.show()

    # Arguments
    if args.open is not None:
        window.tabview.openDocument(args.open)

    # Keep event loop active with timer
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    return app.exec_()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
