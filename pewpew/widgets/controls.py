from PySide2 import QtCore, QtGui, QtWidgets

from typing import Optional

from pewpew.graphics.imageitems import LaserImageItem


class ControlBar(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(256, 32)


class LaserControlBar(ControlBar):
    alphaChanged = QtCore.Signal(float)
    elementChanged = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.toolbar = QtWidgets.QToolBar()

        self.alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(100)
        self.alpha.valueChanged.connect(lambda i: self.alphaChanged.emit(i / 100.0))

        self.elements = QtWidgets.QComboBox()
        self.elements.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.elements.currentTextChanged.connect(self.elementChanged)

        layout = QtWidgets.QHBoxLayout()

        layout.addWidget(self.toolbar, 0)
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel("Alpha:"), 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.alpha, 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.elements, 0, QtCore.Qt.AlignRight)
        self.setLayout(layout)

    def setItem(self, item: LaserImageItem) -> None:
        self.blockSignals(True)

        # Disconnect
        try:
            self.alphaChanged.disconnect()
            self.elementChanged.disconnect()
        except RuntimeError:
            pass

        # Set current values
        self.alpha.setValue(int(item.opacity() * 100.0))

        self.elements.clear()
        self.elements.addItems(item.laser.elements)
        self.elements.setCurrentText(item.element())

        # Connect
        self.alphaChanged.connect(item.setOpacity)
        self.elementChanged.connect(lambda e: [item.setElement(e), item.redraw()])

        self.blockSignals(False)
