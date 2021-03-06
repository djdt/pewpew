import logging

from PySide2 import QtCore, QtGui, QtWidgets

__version__ = "1.2.1"
__loglevel__ = logging.INFO

logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(__loglevel__)

# Set Some Qt attributes
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtGui.QIcon.setThemeName("breeze")
