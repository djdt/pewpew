from PySide2 import QtCore, QtGui, QtWidgets

from pathlib import Path

from typing import Optional, Set, Union

from pewpew.actions import qAction


class OverlayParentItem(QtWidgets.QGraphicsObject):
    def __init__(self, parent: Optional[QtWidgets.QGraphicsItem] = None):
        super().__init__(parent)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        self.setFlag(QtWidgets.QGraphicsItem.ItemHasNoContents)

        self.rect = QtCore.QRectF()
        self.pixmap = None

    def boundingRect(self) -> QtCore.QRectF:
        return self.rect

    def setRect(self, rect: QtCore.QRectF) -> None:
        self.rect = rect

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        paint_requested = any(item.paintRequested() for item in self.childItems())
        if (
            self.pixmap is None
            or self.pixmap.size() != painter.viewport().size()
            or paint_requested
        ):
            self.pixmap = QtGui.QPixmap(painter.viewport().size())
            self.pixmap.fill(QtCore.Qt.transparent)

            pixmap_painter = QtGui.QPainter(self.pixmap)
            for item in self.childItems():
                pixmap_painter.setTransform(self.itemTransform(item)[0].inverted()[0])
                item.setViewport(painter.viewport())
                if item.isVisible():
                    item.paint(pixmap_painter, option, widget)
            pixmap_painter.end()

        painter.save()
        painter.resetTransform()
        painter.drawPixmap(0, 0, self.pixmap)
        painter.restore()


class OverlayItem(QtWidgets.QGraphicsObject):
    def __init__(self, parent: Optional[QtWidgets.QGraphicsItem] = None):
        super().__init__(parent)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        self.setFlag(QtWidgets.QGraphicsItem.ItemHasNoContents)

        self.viewport = QtCore.QRect()
        self.painted = False

    def setViewport(self, rect: QtCore.QRect) -> None:
        self.viewport = rect

    def requestPaint(self) -> None:
        self.painted = False

    def paintRequested(self) -> bool:
        return not self.painted

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        self.painted = True


class OverlayGraphicsView(QtWidgets.QGraphicsView):
    """A graphics view implementing an overlay scene and mouse navigation.

    Updates the overlay pixmap on on view changes.

    Parameters:
        cursors: dict of cursors for interaction modes
        interaction_flags: current interaction modes
    """

    viewScaleChanged = QtCore.Signal()
    viewSizeChanged = QtCore.Signal(QtCore.QRect)

    def __init__(
        self,
        scene: QtWidgets.QGraphicsScene,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(scene, parent)
        self.scene().setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
        self.scene().setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

        self.overlay = OverlayParentItem()
        self.scene().addItem(self.overlay)

        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.cursors = {
            "drag": QtCore.Qt.ClosedHandCursor,
        }
        self.interaction_flags: Set[str] = set()  # Deafult is navigate when empty
        self._last_pos = QtCore.QPoint(0, 0)  # Used for mouse events

    def addOverlayItem(self, item: OverlayItem):
        self.scene().addItem(item)
        item.setParentItem(self.overlay)

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        self.overlay.setPos(rect.topLeft())
        self.overlay.setRect(rect.translated(rect.topLeft()))
        self.overlay.paint(painter, QtWidgets.QStyleOptionGraphicsItem(), None)

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if (
            len(self.interaction_flags) == 0 and event.button() == QtCore.Qt.LeftButton
        ) or event.button() == QtCore.Qt.MiddleButton:
            self.setInteractionFlag("drag")
            self._last_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if "drag" in self.interaction_flags:
            dx = self._last_pos.x() - event.pos().x()
            dy = self._last_pos.y() - event.pos().y()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)
            self._last_pos = event.pos()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if "drag" in self.interaction_flags:
            self.setInteractionFlag("drag", False)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        rect = self.mapFromScene(self.sceneRect()).boundingRect()
        rect.moveTo(0, 0)
        oldrect = QtCore.QRect(QtCore.QPoint(0, 0), event.oldSize())
        if oldrect.contains(rect):
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

        self.viewSizeChanged.emit(self.viewport().rect())

    def saveToFile(self, path: Union[str, Path]) -> None:
        """Save the current view to a file."""
        if isinstance(path, str):
            path = Path(path)

        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self.render(painter)
        pixmap.save(str(path.absolute()))
        painter.end()

    def setInteractionFlag(self, flag: str, on: bool = True) -> None:
        """Update interaction modes and the cursor."""
        if on:
            self.interaction_flags.add(flag)
            if flag in self.cursors:
                self.viewport().setCursor(self.cursors[flag])
        else:
            self.interaction_flags.discard(flag)
            self.viewport().setCursor(QtCore.Qt.ArrowCursor)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        if self.scene() is not None:
            self.scene().invalidate(
                self.mapToScene(self.viewport().rect()).boundingRect(),
                QtWidgets.QGraphicsScene.ForegroundLayer,
            )

    def wheelEvent(self, event: QtGui.QWheelEvent):
        # Save transformation anchor and set to mouse position
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        # Scale a small amount per scroll
        scale = pow(2, event.angleDelta().y() / 360.0)
        self.scale(scale, scale)

        rect = self.mapFromScene(self.sceneRect()).boundingRect()
        if self.viewport().rect().contains(rect):
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.viewScaleChanged.emit()

        self.setTransformationAnchor(anchor)

    def zoomToArea(self, rect: QtCore.QRectF) -> None:
        self.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def zoomReset(self) -> None:
        rect = QtCore.QRectF(0, 0, 0, 0)
        for item in self.scene().items():
            if not isinstance(item, (OverlayItem, OverlayParentItem)):
                rect = rect.united(item.boundingRect())
        self.scene().setSceneRect(rect)
        self.fitInView(rect, QtCore.Qt.KeepAspectRatio)
