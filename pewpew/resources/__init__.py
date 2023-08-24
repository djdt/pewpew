from PySide6 import QtCore, QtGui, QtWidgets

from . import app_icon, icons

# Set Some Qt attributes
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtGui.QIcon.setThemeName("pewpew")
