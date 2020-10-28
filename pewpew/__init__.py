import logging

from PySide2 import QtCore, QtGui, QtWidgets

__version__ = "1.0.6"
__loglevel__ = logging.INFO

logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(__loglevel__)

# Set Some Qt attributes
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtGui.QIcon.setThemeName("breath2")
# Until KDE fix directory viewer
# if sys.platform != "win32":
#     QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DontUseNativeDialogs)
