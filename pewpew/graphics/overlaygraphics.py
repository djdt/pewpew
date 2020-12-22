"""Contains classes used for drawing a static overlay over a view.
"""
from PySide2 import QtCore, QtGui, QtWidgets

from typing import Union, List


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
        self.overlayitems: List[OverlayItem] = []

        self.foreground_pixmap: QtGui.QPixmap = None

    def addOverlayItem(
        self,
        item,
        anchor: QtCore.Qt.AnchorPoint,
        alignment: QtCore.Qt.Alignment = None,
    ):
        item.setFlag(QtWidgets.QGraphicsItem.ItemHasNoContents)
        self.addItem(item)
        self.overlayitems.append(OverlayItem(item, anchor, alignment))

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

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        if self.foreground_pixmap is None:
            self.updateForeground(self.views()[0].viewport().rect())

        painter.save()
        painter.resetTransform()
        # Draw the actual overlay
        painter.drawPixmap(0, 0, self.foreground_pixmap)
        painter.restore()


class OverlayView(QtWidgets.QGraphicsView):
    scaleChanged = QtCore.Signal()
    viewChanged = QtCore.Signal(QtCore.QRect)

    def __init__(
        self,
        scene: OverlayScene,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(scene, parent)
        self.scale_factor = 1.0
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # Only redraw the ForegroundLayer when needed
        self.viewChanged.connect(scene.updateForeground)
        self.scaleChanged.connect(lambda: self.viewChanged.emit(self.viewport().rect()))

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
        scale = 1.0 + event.angleDelta().y() / 360.0
        new_scale = self.sceneScale() * scale

        if new_scale < 1.0:
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.scaleChanged.emit()
        elif 1.0 < new_scale < 100.0:
            self.scale(scale, scale)
            self.scaleChanged.emit()

        self.setTransformationAnchor(anchor)

    def sceneScale(self) -> float:
        return (
            self.mapFromScene(self.sceneRect()).boundingRect().width()
            / self.viewport().width()
        )

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        scale = (
            self.mapFromScene(self.sceneRect()).boundingRect().width()
            / event.oldSize().width()
        )
        if scale <= 1.05:
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

        self.viewChanged.emit(self.viewport().rect())
