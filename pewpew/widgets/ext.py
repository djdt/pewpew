from PySide2 import QtCore, QtGui, QtWidgets

from typing import List, Tuple


class CollapsableWidget(QtWidgets.QWidget):
    def __init__(self, title: str, parent: QtWidgets.QWidget):
        super().__init__(parent)

        self.button = QtWidgets.QToolButton()
        self.button.setArrowType(QtCore.Qt.RightArrow)
        self.button.setAutoRaise(True)
        self.button.setCheckable(True)
        self.button.setChecked(False)
        self.button.setText(title)
        self.button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

        self.line = QtWidgets.QFrame()
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum
        )

        self.area = QtWidgets.QWidget()

        self.button.toggled.connect(self.collapse)

        layout_line = QtWidgets.QHBoxLayout()
        layout_line.addWidget(self.button, 0, QtCore.Qt.AlignLeft)
        layout_line.addWidget(self.line, 1)
        layout_line.setAlignment(QtCore.Qt.AlignTop)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_line, 0)
        layout.addWidget(self.area, 1)
        self.setLayout(layout)

        self.area.hide()

        self.parent().layout().setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

    def collapse(self, down: bool) -> None:
        self.button.setArrowType(QtCore.Qt.DownArrow if down else QtCore.Qt.RightArrow)
        self.area.setVisible(down)
        self.parent().adjustSize()


class MultipleDirDialog(QtWidgets.QFileDialog):
    def __init__(self, parent: QtWidgets.QWidget, title: str, directory: str):
        super().__init__(parent, title, directory)
        self.setFileMode(QtWidgets.QFileDialog.Directory)
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        children = self.findChildren(QtWidgets.QListView)
        children.extend(self.findChildren(QtWidgets.QTreeView))
        for view in children:
            if isinstance(view.model(), QtWidgets.QFileSystemModel):
                view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    @staticmethod
    def getExistingDirectories(
        parent: QtWidgets.QWidget, title: str, directory: str
    ) -> List[str]:
        dlg = MultipleDirDialog(parent, title, directory)
        if dlg.exec():
            return list(dlg.selectedFiles())
        else:
            return []


class RangeSlider(QtWidgets.QSlider):
    value2Changed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setOrientation(QtCore.Qt.Horizontal)

        self._value2 = 99
        self._pressed = False

    def left(self) -> int:
        return min(self.value(), self.value2())

    def setLeft(self, value: int) -> None:
        if self.value() < self._value2:
            self.setValue(value)
        else:
            self.setValue2(value)

    def right(self) -> int:
        return max(self.value(), self.value2())

    def setRight(self, value: int) -> None:
        if self.value() > self._value2:
            self.setValue(value)
        else:
            self.setValue2(value)

    def values(self) -> Tuple[int, int]:
        return self.value(), self.value2()

    def setValues(self, left: int, right: int) -> None:
        self.setValue(left)
        self.setValue2(right)

    def value2(self) -> int:
        return self._value2

    def setValue2(self, value: int) -> None:
        self._value2 = value
        self.value2Changed.emit(self._value2)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, option, QtWidgets.QStyle.SC_SliderGroove, self
        )
        handle = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, option, QtWidgets.QStyle.SC_SliderHandle, self
        )
        pos = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), self.value2(), groove.width()
        )

        handle.moveCenter(QtCore.QPoint(pos, handle.center().y()))
        if handle.contains(event.pos()):
            event.accept()
            self._pressed = True
            self.setSliderDown(True)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pressed:
            option = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(
                QtWidgets.QStyle.CC_Slider,
                option,
                QtWidgets.QStyle.SC_SliderGroove,
                self,
            )
            handle = self.style().subControlRect(
                QtWidgets.QStyle.CC_Slider,
                option,
                QtWidgets.QStyle.SC_SliderHandle,
                self,
            )
            value = self.style().sliderValueFromPosition(
                self.minimum(), self.maximum(), event.pos().x(), groove.width()
            )
            handle.moveCenter(event.pos())
            handle = handle.marginsAdded(
                QtCore.QMargins(
                    handle.width(), handle.width(), handle.width(), handle.width()
                )
            )
            self.setValue2(value)
            self.repaint(handle)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pressed:
            self._pressed = False
            self.setSliderDown(False)
        super().mouseReleaseEvent(event)

    def paintEvent(
        self,
        event: QtGui.QPaintEvent,
    ) -> None:
        painter = QtGui.QPainter(self)
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        option.activeSubControls = QtWidgets.QStyle.SC_None

        if self.isSliderDown():
            option.state |= QtWidgets.QStyle.State_Sunken
            option.activeSubControls = QtWidgets.QStyle.SC_ScrollBarSlider
        else:
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
            option.activeSubControls = self.style().hitTestComplexControl(
                QtWidgets.QStyle.CC_Slider, option, pos, self
            )

        groove = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, option, QtWidgets.QStyle.SC_SliderGroove, self
        )
        start = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), self.left(), groove.width()
        )
        end = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), self.right(), groove.width()
        )

        # Draw grooves
        option.subControls = QtWidgets.QStyle.SC_SliderGroove

        option.sliderPosition = self.maximum() - self.minimum() - self.left()
        option.upsideDown = not option.upsideDown

        cliprect = QtCore.QRect(groove)
        cliprect.setRight(end)
        painter.setClipRegion(QtGui.QRegion(cliprect))

        self.style().drawComplexControl(
            QtWidgets.QStyle.CC_Slider, option, painter, self
        )

        option.upsideDown = not option.upsideDown
        option.sliderPosition = self.right()
        cliprect.setLeft(start)
        cliprect.setRight(groove.right())
        painter.setClipRegion(QtGui.QRegion(cliprect))

        self.style().drawComplexControl(
            QtWidgets.QStyle.CC_Slider, option, painter, self
        )

        painter.setClipRegion(QtGui.QRegion())
        painter.setClipping(False)

        # Draw handles
        option.subControls = QtWidgets.QStyle.SC_SliderHandle

        option.sliderPosition = self.left()
        self.style().drawComplexControl(
            QtWidgets.QStyle.CC_Slider, option, painter, self
        )
        option.sliderPosition = self.right()
        self.style().drawComplexControl(
            QtWidgets.QStyle.CC_Slider, option, painter, self
        )


class ValidColorLineEdit(QtWidgets.QLineEdit):
    def __init__(self, text: str = "", parent: QtWidgets.QWidget = None):
        super().__init__(text, parent)
        self.textChanged.connect(self.revalidate)
        self.color_good = self.palette().color(QtGui.QPalette.Base)
        self.color_bad = QtGui.QColor.fromRgb(255, 172, 172)

    def revalidate(self) -> None:
        self.setValid(self.hasAcceptableInput())

    def setValid(self, valid: bool) -> None:
        palette = self.palette()
        if valid:
            color = self.color_good
        else:
            color = self.color_bad
        palette.setColor(QtGui.QPalette.Base, color)
        self.setPalette(palette)


class ValidColorTextEdit(QtWidgets.QTextEdit):
    def __init__(self, text: str = "", parent: QtWidgets.QWidget = None):
        super().__init__(text, parent)
        self.textChanged.connect(self.revalidate)
        self.color_good = self.palette().color(QtGui.QPalette.Base)
        self.color_bad = QtGui.QColor.fromRgb(255, 172, 172)

    def revalidate(self) -> None:
        self.setValid(self.hasAcceptableInput())

    def setValid(self, valid: bool) -> None:
        palette = self.palette()
        if valid:
            color = self.color_good
        else:
            color = self.color_bad
        palette.setColor(QtGui.QPalette.Base, color)
        self.setPalette(palette)
