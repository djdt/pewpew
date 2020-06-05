import sys
from PySide2 import QtCore, QtWidgets

__version__ = "1.0.1"

# Set Some Qt attributes
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
# Until KDE fix directory viewer
if sys.platform != "win32":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DontUseNativeDialogs)
