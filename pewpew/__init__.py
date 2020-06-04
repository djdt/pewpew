import sys
from PySide2 import QtCore, QtGui, QtWidgets

__version__ = "1.0.1"

from pewpew.mainwindow import MainWindow
from pewpew.resources import app_icon  # noqa: F401
from pewpew.resources import icons  # noqa: F401


def main() -> None:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    if sys.platform != "win32":
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DontUseNativeDialogs)
    QtGui.QIcon.setThemeName("breath")

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
