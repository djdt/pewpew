from PySide2 import QtCore, QtGui, QtWidgets

from pathlib import Path

from pewpew.graphics.overlayitems import OverlayItem

from typing import List, Optional, Set, Union


class OverlayScene(QtWidgets.QGraphicsScene):
    """A graphics scene that also draws OverlayItems in the foreground.

    The forground is saved to a pixmap to limit redrawing.
    """

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(x, y, width, height, parent)
        self.setSortCacheEnabled(True)
        self.setItemIndexMethod(
            QtWidgets.QGraphicsScene.NoIndex
        )  # Turn off BSP indexing, it causes a crash on item removal

        self.overlayitems: List[OverlayItem] = []

        self.foreground_pixmap: Optional[QtGui.QPixmap] = None

    def addOverlayItem(
        self,
        item,
        anchor: QtCore.Qt.AnchorPoint,
        alignment: Optional[QtCore.Qt.Alignment] = None,
    ):
        """Adds an item to the overlay."""
        item.setFlag(
            QtWidgets.QGraphicsItem.ItemHasNoContents
        )  # Drawing handled manually
        self.addItem(item)
        self.overlayitems.append(OverlayItem(item, anchor, alignment))

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        """Draw the foreground pixmap, updates if None."""
        if self.foreground_pixmap is None:
            self.updateForeground(rect)

        painter.save()
        painter.resetTransform()
        # Draw the actual overlay
        painter.drawPixmap(0, 0, self.foreground_pixmap)
        painter.restore()

    def updateForeground(self, rect: QtCore.QRectF) -> None:
        """Update the forground pixmap, rect should equal display size."""
        self.foreground_pixmap = QtGui.QPixmap(rect.size())
        self.foreground_pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(self.foreground_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        for item in self.overlayitems:
            if not item.item.isVisible():
                continue
            transform = QtGui.QTransform()
            transform.translate(item.pos().x(), item.pos().y())
            transform.translate(item.anchorPos(rect).x(), item.anchorPos(rect).y())
            painter.setTransform(transform)
            item.item.paint(painter, QtWidgets.QStyleOptionGraphicsItem(), None)

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        """Intercept events and pass to overlay."""
        view_pos = event.widget().mapFromGlobal(event.screenPos())
        for item in self.overlayitems:
            if item.contains(view_pos, event.widget().rect()):
                item.item.mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        """Intercept events and pass to overlay."""
        view_pos = event.widget().mapFromGlobal(event.screenPos())
        for item in self.overlayitems:
            if item.contains(view_pos, event.widget().rect()):
                item.item.contextMenuEvent(event)
                return
        super().contextMenuEvent(event)


class OverlayView(QtWidgets.QGraphicsView):
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
        scene: OverlayScene,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(scene, parent)
        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.cursors = {
            "drag": QtCore.Qt.ClosedHandCursor,
            "zoom": QtCore.Qt.ArrowCursor,
        }
        self.interaction_flags: Set[str] = set()  # Deafult is navigate when empty
        self._last_pos = QtCore.QPoint(0, 0)  # Used for mouse events

        # Only redraw the ForegroundLayer when needed
        self.viewSizeChanged.connect(self.updateForeground)
        self.viewScaleChanged.connect(self.updateForeground)

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if "zoom" in self.interaction_flags and event.button() == QtCore.Qt.LeftButton:
            self._last_pos = event.pos()
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        elif (
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
        if "zoom" in self.interaction_flags:
            self.setInteractionFlag("zoom", False)
            rect = QtCore.QRect(self._last_pos, event.pos())
            rect = self.mapToScene(rect).boundingRect()
            self.zoomToArea(rect)
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
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

    def updateForeground(self, rect: Optional[QtCore.QRect] = None) -> None:
        if rect is None:
            rect = self.viewport().rect()
        self.scene().updateForeground(rect)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        # Save transformation anchor and set to mouse position
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        rect = self.mapFromScene(self.sceneRect()).boundingRect()

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
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
