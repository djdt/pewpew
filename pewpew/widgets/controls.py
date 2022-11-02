from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.graphics.imageitems import LaserImageItem


class ControlBar(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.toolbar = QtWidgets.QToolBar()

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.toolbar, 0)
        layout.addStretch(1)
        self.setLayout(layout)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(256, 32)

    def locked(self) -> bool:
        return False


class LaserControlBar(ControlBar):
    alphaChanged = QtCore.Signal(float)
    elementChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(100)
        self.alpha.valueChanged.connect(lambda i: self.alphaChanged.emit(i / 100.0))

        self.elements = QtWidgets.QComboBox()
        self.elements.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.elements.currentTextChanged.connect(self.elementChanged)

        self.layout().addWidget(QtWidgets.QLabel("Alpha:"), 0, QtCore.Qt.AlignRight)
        self.layout().addWidget(self.alpha, 0, QtCore.Qt.AlignRight)
        self.layout().addWidget(self.elements, 0, QtCore.Qt.AlignRight)

    def setItem(self, item: LaserImageItem) -> None:
        self.blockSignals(True)

        # Disconnect throws a RuntimeError if not connected...
        for signal in [self.alphaChanged, self.elementChanged]:
            try:
                signal.disconnect()
            except RuntimeError:
                pass

        # Set current values
        self.alpha.setValue(int(item.opacity() * 100.0))

        self.elements.clear()
        self.elements.addItems(item.laser.elements)
        self.elements.setCurrentText(item.element())

        # Connect
        self.alphaChanged.connect(item.setOpacity)
        self.on_element_changed = lambda e: [item.setElement(e), item.redraw()]
        self.elementChanged.connect(self.on_element_changed)

        self.blockSignals(False)


class ImageControlBar(ControlBar):
    alphaChanged = QtCore.Signal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(100)
        self.alpha.valueChanged.connect(lambda i: self.alphaChanged.emit(i / 100.0))

        self.layout().addWidget(QtWidgets.QLabel("Alpha:"), 0, QtCore.Qt.AlignRight)
        self.layout().addWidget(self.alpha, 0, QtCore.Qt.AlignRight)

    def setItem(self, item: LaserImageItem) -> None:
        self.blockSignals(True)

        # Disconnect throws a RuntimeError if not connected...
        try:
            self.alphaChanged.disconnect()
        except RuntimeError:
            pass

        # Set current values
        self.alpha.setValue(int(item.opacity() * 100.0))

        # Connect
        self.alphaChanged.connect(item.setOpacity)

        self.blockSignals(False)
