from PyQt5 import QtCore, QtWidgets

from util.laser import LaserParams


class Controls(QtWidgets.QWidget):
    EDITABLE_PARAMS = ['spotsize', 'speed', 'scantime', 'gradient',
                       'intercept']
    def __init__(self, parent=None, params=LaserParams()):
        super().__init__(parent)

        form = QtWidgets.QFormLayout()

        for p in Controls.EDITABLE_PARAMS:
            le = QtWidgets.QLineEdit(str(getattr(params, p, 0.0)))
            form.addRow(p.capitalize() + ":", le)
            setattr(self, p + "LineEdit", le)

        self.updateButton = QtWidgets.QPushButton("&Update")
        self.updateAllButton = QtWidgets.QPushButton("Update &All")

        self.setLayout(form)

    def updateParams(self, params):
        for p in Controls.EDITABLE_PARAMS:
            value = params[p]
            le = getattr(self, p + "LineEdit")
            le.setText(str(value))
