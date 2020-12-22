"""This is a more efficient way (less paints get called.)"""
from PySide2 import QtWidgets

import sys

from pewlib import io

from viewoptions import ViewOptions
from laserimageview import LaserImageView


app = QtWidgets.QApplication()

laser = io.npz.load(sys.argv[1])

data = laser.get(laser.isotopes[0])

view = LaserImageView(ViewOptions())
view.drawData(data, laser.isotopes[0])

mainwindow = QtWidgets.QMainWindow()

widget = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout()
layout.addWidget(view)
widget.setLayout(layout)

mainwindow.setCentralWidget(widget)
mainwindow.resize(1280, 800)
mainwindow.show()

app.exec_()
