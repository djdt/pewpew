from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.actions import qAction, qToolButton
from pewpew.graphics.imageitems import (
    LaserImageItem,
    RGBLaserImageItem,
    ScaledImageItem,
)
from pewpew.widgets.dialogs import NameEditDialog


class EditComboBox(QtWidgets.QComboBox):
    """Combo box with a context menu for editing names."""

    namesSelected = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.action_edit_names = qAction(
            "document-edit",
            "Edit Names",
            "Edit image names.",
            self.actionNameEditDialog,
        )

    def actionNameEditDialog(self) -> QtWidgets.QDialog:
        names = [self.itemText(i) for i in range(self.count())]
        dlg = NameEditDialog(names, allow_remove=True, parent=self)
        dlg.namesSelected.connect(self.namesSelected)
        dlg.open()
        return dlg

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        event.accept()
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_edit_names)
        menu.popup(event.globalPos())


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

        self.elements = EditComboBox()
        self.elements.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.elements.currentTextChanged.connect(self.elementChanged)

        self.layout().addWidget(QtWidgets.QLabel("Alpha:"), 0, QtCore.Qt.AlignRight)
        self.layout().addWidget(self.alpha, 0, QtCore.Qt.AlignRight)
        self.layout().addWidget(self.elements, 0, QtCore.Qt.AlignRight)

    def setItem(self, item: LaserImageItem) -> None:
        self.blockSignals(True)

        # Disconnect throws a RuntimeError if not connected...
        for signal in [
            self.alphaChanged,
            self.elementChanged,
            self.elements.namesSelected,
        ]:
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
        self.elementChanged.connect(item.setElement)
        self.on_rename = lambda e: [item.renameElements(e), self.setItem(item)]
        self.elements.namesSelected.connect(self.on_rename)

        self.blockSignals(False)


class RGBLaserControl(QtWidgets.QWidget):
    alphaChanged = QtCore.Signal(float)
    elementChanged = QtCore.Signal(str)
    colorChanged = QtCore.Signal(QtGui.QColor)
    visibilityChanged = QtCore.Signal(bool)

    def __init__(self, color: QtGui.QColor, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.action_color = qAction(
            "color-picker",
            "Color",
            "Select the color for this element.",
            self.selectColor,
        )

        # self.alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # self.alpha.setRange(0, 100)
        # self.alpha.setValue(100)
        # self.alpha.valueChanged.connect(lambda i: self.alphaChanged.emit(i / 100.0))

        self.elements = QtWidgets.QComboBox()
        self.elements.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.elements.currentTextChanged.connect(self.elementChanged)

        self.button_color = qToolButton(action=self.action_color)

        self.effect_color = QtWidgets.QGraphicsColorizeEffect()
        self.effect_color.setColor(color)
        self.button_color.setGraphicsEffect(self.effect_color)

        # self.layout().addWidget(QtWidgets.QLabel("Alpha:"), 0, QtCore.Qt.AlignRight)
        # self.layout().addWidget(self.alpha, 0, QtCore.Qt.AlignRight)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.elements, 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.button_color, 0)
        self.setLayout(layout)

    def getColor(self) -> QtGui.QColor:
        return self.effect_color.color()

    def setColor(self, color: QtGui.QColor) -> None:
        if color != self.effect_color.color():
            self.effect_color.setColor(color)
            self.colorChanged.emit(color)

    def selectColor(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QColorDialog(self.getColor(), self)
        dlg.colorSelected.connect(self.setColor)
        dlg.open()
        return dlg


class RGBLaserControlBar(ControlBar):
    alphaChanged = QtCore.Signal(float)
    colorsChanged = QtCore.Signal(list)
    elementsChanged = QtCore.Signal(list)

    def __init__(self, color: QtGui.QColor, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(100)
        self.alpha.valueChanged.connect(lambda i: self.alphaChanged.emit(i / 100.0))

        self.controls = [
            RGBLaserControl(color)
            for color in [
                QtGui.QColor(255, 0, 0),
                QtGui.QColor(0, 255, 0),
                QtGui.QColor(0, 0, 255),
            ]
        ]
        for control in self.controls:
            control.elementChanged.connect(self.onElementChanged)
            control.colorChanged.connect(self.onColorChanged)

        self.layout().addWidget(QtWidgets.QLabel("Alpha:"), 0, QtCore.Qt.AlignRight)
        self.layout().addWidget(self.alpha, 0, QtCore.Qt.AlignRight)
        for control in self.controls:
            self.layout().addWidget(control, 0, QtCore.Qt.AlignRight)

    def onColorChanged(self) -> None:
        self.colorsChanged.emit([c.getColor() for c in self.controls])

    def onElementChanged(self) -> None:
        texts = [c.elements.currentText() for c in self.controls]
        self.elementsChanged.emit([text for text in texts if len(text) > 0])

    def setItem(self, item: RGBLaserImageItem) -> None:
        self.blockSignals(True)

        # Disconnect throws a RuntimeError if not connected...
        for signal in [
            self.alphaChanged,
            self.elementChanged,
            self.elements.namesSelected,
        ]:
            try:
                signal.disconnect()
            except RuntimeError:
                pass

        # Set current values
        self.alpha.setValue(int(item.opacity() * 100.0))

        for color, control in zip(item.elementColors(), self.controls):
            control.setColor(color)

        for control in self.controls:
            control.elements.clear()
            control.elements.addItems(item.laser.elements)

        for element, control in zip(item.elements(), self.controls):
            control.elements.setCurrentText(element)

        for element, control in zip(item.elements(), self.controls):
            control.elements.setCurrentText(element)

        # Connect
        self.alphaChanged.connect(item.setOpacity)
        self.colorsChanged.connect(item.setElementColors)
        self.elementsChanged.connect(item.setElements)
        # self.on_rename = lambda e: [item.renameElements(e), self.setItem(item)]
        # self.elements.namesSelected.connect(self.on_rename)

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

    def setItem(self, item: ScaledImageItem) -> None:
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
