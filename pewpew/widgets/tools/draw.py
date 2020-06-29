import copy
import numpy as np
from PySide2 import QtCore, QtWidgets
from matplotlib.backend_bases import KeyEvent, MouseEvent, LocationEvent
from matplotlib.cm import get_cmap

from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import InteractiveCanvas, LaserCanvas

from typing import List, Tuple


class DrawUndoState(object):
    def __init__(self, pos: Tuple[int, int], data: np.ndarray):
        self.data = data
        self.x1, self.y1 = pos
        self.x2, self.y2 = data.shape + np.array(pos)

    def undo(self, x: np.ndarray) -> None:
        x[self.x1 : self.x2, self.y1 : self.y2] = self.data


class DrawCanvas(LaserCanvas, InteractiveCanvas):
    def __init__(
        self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None
    ) -> None:
        super().__init__(viewoptions=viewoptions, parent=parent)
        self.state = set(["move"])
        self.brush_button = 1
        self.move_button = 2

        self.brush = {"shape": None, "size": 1, "value": np.nan}

        self.undo_states: List[DrawUndoState] = []

    def redrawFigure(self) -> None:
        super().redrawFigure()

    def drawData(
        self,
        data: np.ndarray,
        extent: Tuple[float, float, float, float],
        isotope: str = None,
    ) -> None:
        if self.image is not None:
            self.image.remove()

        # Calculate the range
        vmin, vmax = self.viewoptions.colors.get_range_as_float(isotope, data)
        # Ensure than vmin is actually lower than vmax
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        # Set nan's to be pink
        if isinstance(self.viewoptions.image.cmap, str):
            cmap = get_cmap(self.viewoptions.image.cmap)
        else:
            cmap = copy.copy(self.viewoptions.image.cmap)
        cmap.set_bad(color="pink")

        # Plot the image
        self.image = self.ax.imshow(
            data,
            cmap=cmap,
            interpolation=self.viewoptions.image.interpolation,
            alpha=self.viewoptions.image.alpha,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            origin="upper",
        )
        # Rescale to full image
        self.view_limits = extent

    def ignore_event(self, event: LocationEvent) -> bool:
        if event.name in ["key_press_event"]:
            return True

        return super().ignore_event(event)

    def press(self, event: MouseEvent) -> None:
        if event.button == self.brush_button:
            x, y = event.xdata, event.ydata
            shape = self.image.get_array().shape
            # Snap to pixels
            pxy = 1.0 / np.array([shape[1], shape[0]])
            x, y = pxy * np.round([x, y] / pxy)
            if self.brush["shape"] in ["square", "circle"]:
                data = self.image.get_array()
                pass
            elif self.brush["shape"] == "fill":
                pass

    def getBrushMask(self, x: int, y: int) -> None:
        mask = np.zeros(self.image.get_array().shape, dtype=bool)
        size = self.brush["size"]
        if self.brush["shape"] == "circle":
            xx, yy = np.mgrisize[:size, :size]
            r = size // 2
            circle = ((xx - r) ** 2 + (yy - r) ** 2) < r ** 2
            mask[x - r : x + size, y - r, y + size] = circle
        elif self.brush["shape"] == "square":
            mask[x - r : x + size, y - r : y + size] = True
        elif self.brush["shape"] == "bucket":
            value = self.image.get_array()[x, y]
            # Get connected that have diff less than size

        return mask


    def release(self, event: MouseEvent) -> None:
        pass

    def move(self, event: MouseEvent) -> None:
        if (
            all(state in self.state for state in ["move", "zoom"])
            # and "selection" not in self.state
            and event.button == self.move_button
        ):
            x1, x2, y1, y2 = self.view_limits
            xmin, xmax, ymin, ymax = self.extent
            dx = self.eventpress.xdata - event.xdata
            dy = self.eventpress.ydata - event.ydata

            # Move in opposite direction to drag
            if x1 + dx > xmin and x2 + dx < xmax:
                x1 += dx
                x2 += dx
            if y1 + dy > ymin and y2 + dy < ymax:
                y1 += dy
                y2 += dy
            self.view_limits = x1, x2, y1, y2

        # Update the status bar
        x, y = event.xdata, event.ydata
        v = self.image.get_cursor_data(event)
        self.cursorMoved.emit(x, y, v)

    def scroll(self, event: MouseEvent) -> None:
        zoom_factor = 0.1 * event.step

        x1, x2, y1, y2 = self.view_limits

        x1 = x1 + (event.xdata - x1) * zoom_factor
        x2 = x2 - (x2 - event.xdata) * zoom_factor
        y1 = y1 + (event.ydata - y1) * zoom_factor
        y2 = y2 - (y2 - event.ydata) * zoom_factor

        if x1 > x2 or y1 > y2:
            return

        xmin, xmax, ymin, ymax = self.extent

        # If (un)zoom overlaps an edge attempt to shift it
        if x1 < xmin:
            x1, x2 = xmin, min(xmax, x2 + (xmin - x1))
        if x2 > xmax:
            x1, x2 = max(xmin, x1 - (x2 - xmax)), xmax

        if y1 < ymin:
            y1, y2 = ymin, min(ymax, y2 + (ymin - y1))
        if y2 > ymax:
            y1, y2 = max(ymin, y1 - (y2 - ymax)), ymax

        if (x1, x2, y1, y2) != self.extent:
            self.state.add("zoom")
        else:
            self.state.discard("zoom")
        self.view_limits = x1, x2, y1, y2

    def axes_enter(self, event: LocationEvent) -> None:
        pass

    def axes_leave(self, event: LocationEvent) -> None:
        self.cursorClear.emit()
