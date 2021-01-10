"""Contains classes used for drawing a static overlay over a view.
"""
from PySide2 import QtCore, QtGui, QtWidgets

from pathlib import Path

from typing import List, Set, Union


class OverlayItem(object):
    def __init__(
        self,
        item: QtWidgets.QGraphicsItem,
        anchor: Union[QtCore.Qt.AnchorPoint, QtCore.Qt.Corner] = None,
        alignment: QtCore.Qt.Alignment = None,
    ):
        if anchor is None:
            anchor = QtCore.Qt.TopLeftCorner
        if alignment is None:
            alignment = QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft

        self.item = item
        self.anchor = anchor
        self.alignment = alignment

    def anchorPos(self, rect: QtCore.QRectF) -> QtCore.QPointF:
        if isinstance(self.anchor, QtCore.Qt.Corner):
            if self.anchor == QtCore.Qt.TopLeftCorner:
                pos = rect.topLeft()
            elif self.anchor == QtCore.Qt.TopRightCorner:
                pos = rect.topRight()
            elif self.anchor == QtCore.Qt.BottomLeftCorner:
                pos = rect.bottomLeft()
            else:  # BottomRightCorner
                pos = rect.bottomRight()
        else:  # AnchorPoint
            if self.anchor == QtCore.Qt.AnchorTop:
                pos = QtCore.QPointF(rect.center().x(), rect.top())
            elif self.anchor == QtCore.Qt.AnchorLeft:
                pos = QtCore.QPointF(rect.left(), rect.center().y())
            elif self.anchor == QtCore.Qt.AnchorRight:
                pos = QtCore.QPointF(rect.right(), rect.center().y())
            elif self.anchor == QtCore.Qt.AnchorBottom:
                pos = QtCore.QPointF(rect.center().x(), rect.bottom())
            else:
                raise ValueError("Only Top, Left, Right, Bottom anchors supported.")

        return pos

    def pos(self) -> QtCore.QPointF:
        pos = self.item.pos()  # Aligned Left and Top
        rect = self.item.boundingRect()

        if self.alignment & QtCore.Qt.AlignHCenter:
            pos.setX(pos.x() - rect.width() / 2.0)
        elif self.alignment & QtCore.Qt.AlignRight:
            pos.setX(pos.x() - rect.width())

        if self.alignment & QtCore.Qt.AlignVCenter:
            pos.setY(pos.y() - rect.height() / 2.0)
        elif self.alignment & QtCore.Qt.AlignBottom:
            pos.setY(pos.y() - rect.height())

        return pos

    def contains(self, view_pos: QtCore.QPoint, view_rect: QtCore.QRect) -> bool:
        rect = self.item.boundingRect()
        rect.moveTo(self.pos() + self.anchorPos(view_rect))
        return rect.contains(view_pos)


class OverlayScene(QtWidgets.QGraphicsScene):
    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(x, y, width, height, parent)
        self.setSortCacheEnabled(True)
        self.setItemIndexMethod(
            QtWidgets.QGraphicsScene.NoIndex
        )  # Turn off BSP indexing, it causes a crash on item removal

        self.overlayitems: List[OverlayItem] = []

        self.foreground_pixmap: QtGui.QPixmap = None

    def addOverlayItem(
        self,
        item,
        anchor: QtCore.Qt.AnchorPoint,
        alignment: QtCore.Qt.Alignment = None,
    ):
        item.setFlag(
            QtWidgets.QGraphicsItem.ItemHasNoContents
        )  # Drawing handled manually
        self.addItem(item)
        self.overlayitems.append(OverlayItem(item, anchor, alignment))

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRect):
        if self.foreground_pixmap is None:
            self.updateForeground(rect)

        painter.save()
        painter.resetTransform()
        # Draw the actual overlay
        painter.drawPixmap(0, 0, self.foreground_pixmap)
        painter.restore()

    def updateForeground(self, rect: QtCore.QRect) -> None:
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

            # painter.setBrush(QtGui.QBrush(QtCore.Qt.red, QtCore.Qt.Dense7Pattern))
            # painter.drawRect(item.item.boundingRect())

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        view_pos = event.widget().mapFromGlobal(event.screenPos())
        for item in self.overlayitems:
            if item.contains(view_pos, event.widget().rect()):
                item.item.mouseDoubleClickEvent(event)


class OverlayView(QtWidgets.QGraphicsView):
    viewScaleChanged = QtCore.Signal()
    viewSizeChanged = QtCore.Signal(QtCore.QRect)

    def __init__(
        self,
        scene: OverlayScene,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(scene, parent)
        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.cursors = {
            "navigate": QtCore.Qt.ArrowCursor,
            "drag": QtCore.Qt.ClosedHandCursor,
            "zoom": QtCore.Qt.ArrowCursor,
        }
        self.interaction_mode = "navigate"
        self.interaction_flags: Set[str] = set()
        self._last_pos = QtCore.QPoint(0, 0)  # Used for mouse events

        # Only redraw the ForegroundLayer when needed
        self.viewSizeChanged.connect(self.updateForeground)
        self.viewScaleChanged.connect(self.updateForeground)

    def copyToClipboard(self) -> None:
        QtWidgets.QApplication.clipboard().setPixmap(self.grab(self.viewport().rect()))

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if "zoom" in self.interaction_flags and event.button() == QtCore.Qt.LeftButton:
            self._last_pos = event.pos()
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
            super().mousePressEvent(event)
        elif (
            self.interaction_mode == "navigate"
            and event.button() == QtCore.Qt.LeftButton
        ) or event.button() == QtCore.Qt.MiddleButton:
            self.setInteractionFlag("drag")
            self.viewport().setCursor(QtCore.Qt.ClosedHandCursor)
            self._last_pos = event.pos()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if "drag" in self.interaction_flags:
            last = self.mapToGlobal(self._last_pos)
            dx = last.x() - event.globalPos().x()
            dy = last.y() - event.globalPos().y()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)
            self._last_pos = event.pos()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if "drag" in self.interaction_flags:
            self.setInteractionFlag("drag", False)
            # Restore cursor
            self.viewport().setCursor(
                self.cursors.get(self.interaction_mode, QtCore.Qt.ArrowCursor)
            )
        elif "zoom" in self.interaction_flags:
            self.setInteractionFlag("zoom", False)
            rect = QtCore.QRect(self._last_pos, event.pos())
            rect = self.mapToScene(rect).boundingRect()
            self.zoomToArea(rect)
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            super().mouseReleaseEvent(event)
        else:
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
        if isinstance(path, str):
            path = Path(path)

        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self.render(painter)
        pixmap.save(str(path.absolute()))
        painter.end()

    def setInteractionFlag(self, flag: str, on: bool = True) -> None:
        if on:
            self.interaction_flags.add(flag)
        else:
            self.interaction_flags.discard(flag)

    def setInteractionMode(self, mode: str) -> None:
        self.viewport().setCursor(self.cursors.get(mode, QtCore.Qt.ArrowCursor))
        self.interaction_mode = mode

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        if self.scene() is not None:
            self.scene().invalidate(
                self.mapToScene(self.viewport().rect()).boundingRect(),
                QtWidgets.QGraphicsScene.ForegroundLayer,
            )

    def updateForeground(self, rect: QtCore.QRect = None) -> None:
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
