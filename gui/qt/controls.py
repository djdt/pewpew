from PyQt5 import QtCore, QtWidgets

from util.laser import LaserParams


class Controls(QtWidgets.QWidget):
    def __init__(self, parent=None, params=LaserParams()):
        super().__init__(parent)
        self.resize(100, 100)

        self.spotsizeLE = QtWidgets.QLineEdit(str(params.spotsize))
        self.speedLE = QtWidgets.QLineEdit(str(params.speed))
        self.scantimeLE = QtWidgets.QLineEdit(str(params.scantime))

        form = QtWidgets.QFormLayout()

        for p in ['spotsize', 'speed', 'scantime', 'gradient', 'intercept']:
            line_edit = QtWidgets.QLineEdit(str(getattr(params, p, 0.0)))
            form.addRow(p.capitalize() + ":", line_edit)
            setattr(self, p + "LineEdit", line_edit)

        self.setLayout(form)

    def updateParams(self, params):
        pass
