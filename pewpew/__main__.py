import sys
from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.mainwindow import MainWindow
from pewpew.resources import app_icon  # noqa: F401
from pewpew.resources import icons  # noqa: F401


QtGui.QIcon.setThemeName("breath")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    sys.excepthook = window.exceptHook
    window.show()
    window.setWindowIcon(QtGui.QIcon(":/app.ico"))

    # Keep event loop active with timer
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    app.exec_()


if __name__ == "__main__":
    main()
