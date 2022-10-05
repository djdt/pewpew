import logging

from PySide6 import QtCore, QtGui, QtWidgets

__version__ = "1.3.3"
__loglevel__ = logging.DEBUG

logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(__loglevel__)

# Set Some Qt attributes
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtGui.QIcon.setThemeName("breeze")
