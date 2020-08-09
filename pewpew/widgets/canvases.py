import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent, LocationEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage, imsave
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import AxesWidget

from mpl_toolkits.axes_grid1 import make_axes_locatable

from pew.laser import Laser

from pewpew.lib.mpltools import MetricSizeBar
from pewpew.lib.viewoptions import ViewOptions

from typing import Callable, List, Tuple


class BasicCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        figsize: Tuple[float, float] = (5.0, 5.0),
        parent: QtWidgets.QWidget = None,
    ):
        fig = Figure(dpi=100, frameon=False, tight_layout=True, figsize=figsize)
        super().__init__(fig)
        self.ax: Axes = None

        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        action_copy_image = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy Image To Clipboard", self
        )
        action_copy_image.setStatusTip("Copies the canvas to the clipboard.")
        action_copy_image.triggered.connect(self.copyToClipboard)

        context_menu = QtWidgets.QMenu(self.parent())
        context_menu.addAction(action_copy_image)
        context_menu.popup(event.globalPos())

    def drawFigure(self) -> None:
        pass

    def copyToClipboard(self) -> None:
        bbox = (
            self.figure.get_tightbbox(self.get_renderer())
            .transformed(self.figure.dpi_scale_trans)
            .padded(5)  # Pad to look nicer
        )
        (x0, y0), (x1, y1) = bbox.get_points().astype(int)
        ymax = self.size().height()  # We need to invert for mpl to Qt
        QtWidgets.QApplication.clipboard().setPixmap(
            self.grab(QtCore.QRect(x0, ymax - y1, x1 - x0, y1 - y0))
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(250, 250)


class ImageCanvas(BasicCanvas):
    def __init__(
        self,
        figsize: Tuple[float, float] = (5.0, 5.0),
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(figsize, parent)
        self.image: AxesImage = None

    def drawFigure(self) -> None:
        view_limits = self.view_limits if self.ax is not None else None

        self.figure.clear()
        self.ax = self.figure.add_subplot(
            facecolor="black", autoscale_on=False, xmargin=0, ymargin=0
        )
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        if view_limits is not None:
            self.view_limits = view_limits

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        if self.image is None:
            return (0.0, 0.0, 0.0, 0.0)
        else:
            return self.image.get_extent()

    @property
    def view_limits(self) -> Tuple[float, float, float, float]:
        x0, x1, = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        return x0, x1, y0, y1

    @view_limits.setter
    def view_limits(self, limits: Tuple[float, float, float, float]) -> None:
        x0, x1, y0, y1 = limits
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.draw_idle()

    def extentForAspect(
        self, extent: Tuple[float, float, float, float], aspect: float = None
    ) -> Tuple[float, float, float, float]:
        # Breaks on higher aspect
        if aspect is None:
            aspect = self.width() / self.height()

        x0, x1, y0, y1 = extent
        width, height = x1 - x0, y1 - y0

        if width > 0.0 and height > 0.0 and width / height >= aspect:
            y0, y1 = (
                y0 + (height - width / aspect) / 2.0,
                y0 + (height + width / aspect) / 2.0,
            )
        else:
            x0, x1 = (
                x0 + (width - height * aspect) / 2.0,
                x0 + (width + height * aspect) / 2.0,
            )
        return x0, x1, y0, y1

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)

        x0, x1, y0, y1 = self.view_limits
        xmin, xmax, ymin, ymax = self.extentForAspect(self.extent)

        rw = event.size().width() / event.oldSize().width()
        rh = event.size().height() / event.oldSize().height()

        # Check if view limits for the full aspect extent of previous size
        if np.allclose(
            self.view_limits,
            self.extentForAspect(
                self.extent, aspect=event.oldSize().width() / event.oldSize().height()
            ),
        ):
            # Set view limits at new full aspect extent
            x0, x1, y0, y1 = xmin, xmax, ymin, ymax
        else:  # Adjust the view limits to fit aspect
            w, h = (x1 - x0), (y1 - y0)
            x0, x1 = x0 + w / 2.0 - (w / 2.0) * rw, x0 + w / 2.0 + (w / 2.0) * rw
            x0, x1 = max(xmin, x0), min(xmax, x1)
            y0, y1 = y0 + h / 2.0 - (h / 2.0) * rh, y0 + h / 2.0 + (h / 2.0) * rh
            y0, y1 = max(ymin, y0), min(ymax, y1)

        self.view_limits = (x0, x1, y0, y1)

    def saveRawImage(self, path: str, pixel_size: int = 1) -> None:
        vmin, vmax = self.image.get_clim()

        imsave(
            path,
            self.image.get_array(),
            vmin=vmin,
            vmax=vmax,
            cmap=self.image.cmap,
            origin=self.image.origin,
            dpi=100,
        )


class InteractiveImageCanvas(ImageCanvas):
    cursorClear = QtCore.Signal()
    cursorMoved = QtCore.Signal(float, float, float)

    def __init__(
        self,
        figsize: Tuple[float, float] = (5.0, 5.0),
        move_button: int = 0,
        widget_button: int = 0,
        state: Tuple[str, ...] = ("move", "scroll"),
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(figsize, parent)

        self.cids: List[int] = []
        self.default_events = {
            "axes_enter_event": self._axes_enter,
            "axes_leave_event": self._axes_leave,
            "button_press_event": self._press,
            "button_release_event": self._release,
            "key_press_event": self._keypress,
            "motion_notify_event": self._move,
            "scroll_event": self._scroll,
        }

        for event, callback in self.default_events.items():
            self.connect_event(event, callback)

        self.state = set(state)

        self.move_button = move_button
        self.widget_button = widget_button

        self.widget: AxesWidget = None

    def drawFigure(self) -> None:
        super().drawFigure()
        # Update widget ax
        if self.widget is not None:
            self.widget.ax = self.ax

    def close(self) -> None:
        self.disconnect_events()
        super().close()

    def connect_event(self, event: str, callback: Callable) -> None:
        self.cids.append(self.mpl_connect(event, callback))

    def disconnect_events(self) -> None:
        for cid in self.cids:
            self.mpl_disconnect(cid)
        self.cids.clear()

    def ignore_event(self, event: MouseEvent) -> bool:
        if hasattr(event, "inaxes") and event.inaxes != self.ax:
            return True
        if self.widget is not None and self.widget.get_active():
            return True
        return False

    def _axes_enter(self, event: LocationEvent) -> None:
        if self.ignore_event(event):
            return
        self.axes_enter(event)

    def axes_enter(self, event: LocationEvent) -> None:
        pass

    def _axes_leave(self, event: LocationEvent) -> None:
        if self.ignore_event(event):
            return
        self.axes_leave(event)

    def axes_leave(self, event: LocationEvent) -> None:
        self.cursorClear.emit()

    def _press(self, event: MouseEvent) -> None:
        if self.ignore_event(event):
            return
        self.eventpress = event
        self.press(event)

    def press(self, event: MouseEvent) -> None:
        pass

    def _release(self, event: MouseEvent) -> None:
        if self.ignore_event(event):
            return
        self.eventrelease = event
        self.release(event)

    def release(self, event: MouseEvent) -> None:
        pass

    def _keypress(self, event: KeyEvent) -> None:
        if self.ignore_event(event):
            return
        self.keypress(event)

    def keypress(self, event: KeyEvent) -> None:
        pass

    def _move(self, event: MouseEvent) -> None:
        if self.ignore_event(event):
            return
        self.move(event)
        self.moveCursor(event)

    def move(self, event: MouseEvent) -> None:
        if (
            all(state in self.state for state in ["move", "zoom"])
            and event.button == self.move_button
        ):
            x1, x2, y1, y2 = self.view_limits
            xmin, xmax, ymin, ymax = self.extentForAspect(self.extent)
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

    def moveCursor(self, event: MouseEvent) -> None:
        # contains() returns 2 args, bool and dict
        if self.image is not None and self.image.contains(event)[0]:
            # Update the status bar
            x, y = event.xdata, event.ydata
            v = self.image.get_cursor_data(event)
            self.cursorMoved.emit(x, y, v)
        else:
            self.cursorClear.emit()

    def _scroll(self, event: MouseEvent) -> None:
        if self.ignore_event(event):
            return
        self.scroll(event)

    def scroll(self, event: MouseEvent) -> None:
        if "scroll" not in self.state:
            return
        zoom_factor = 0.1 * event.step

        x1, x2, y1, y2 = self.view_limits

        x1 = x1 + (event.xdata - x1) * zoom_factor
        x2 = x2 - (x2 - event.xdata) * zoom_factor
        y1 = y1 + (event.ydata - y1) * zoom_factor
        y2 = y2 - (y2 - event.ydata) * zoom_factor

        if x1 > x2 or y1 > y2:
            return

        xmin, xmax, ymin, ymax = self.extentForAspect(self.extent)

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

    def zoom(self, press: MouseEvent, release: MouseEvent) -> None:
        self.widget = None
        x0, x1, y0, y1 = press.xdata, release.xdata, press.ydata, release.ydata
        xmin, xmax, ymin, ymax = self.extent
        x0, x1 = max(xmin, x0), min(xmax, x1)
        y0, y1 = max(ymin, y0), min(ymax, y1)

        self.view_limits = self.extentForAspect((x0, x1, y0, y1))
        self.state.add("zoom")

    def unzoom(self) -> None:
        self.view_limits = self.extentForAspect(self.extent)
        self.state.discard("zoom")


class SelectableImageCanvas(InteractiveImageCanvas):
    def __init__(
        self,
        move_button: int = 0,
        widget_button: int = 0,
        selection_rgba: Tuple[int, int, int, int] = (255, 0, 0, 255),
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(
            move_button=move_button, widget_button=widget_button, parent=parent
        )

        self.selection: np.ndarray = None
        self.selection_image: AxesImage = None
        self.selection_rgba = np.array(selection_rgba, dtype=np.uint8)

    def drawSelection(self) -> None:
        if self.selection_image is not None:
            self.selection_image.remove()
            self.selection_image = None

        # Selection is empty
        if self.selection is None or np.all(self.selection == 0):
            return

        assert self.selection.dtype == np.bool
        assert self.image is not None

        image = np.zeros((*self.selection.shape[:2], 4), dtype=np.uint8)
        image[self.selection] = self.selection_rgba
        self.selection_image = self.ax.imshow(
            image,
            extent=self.image.get_extent(),
            transform=self.image.get_transform(),
            interpolation="none",
            origin=self.image.origin,
        )

    def clearSelection(self) -> None:
        if self.widget is not None:
            self.widget.set_active(False)
            self.widget.set_visible(False)
            self.widget.update()
        self.state.discard("selection")
        self.selection = None
        self.widget = None
        self.drawSelection()
        self.draw_idle()

    def getMaskedData(self) -> np.ndarray:
        data = self.image.get_array()
        mask = self.selection
        if mask is not None and not np.all(mask):
            ix, iy = np.nonzero(mask)
            x0, x1, y0, y1 = np.min(ix), np.max(ix) + 1, np.min(iy), np.max(iy) + 1
            data = np.where(mask[x0:x1, y0:y1], data[x0:x1, y0:y1], np.nan)
        return data


class LaserImageCanvas(SelectableImageCanvas):
    def __init__(
        self,
        viewoptions: ViewOptions,
        move_button: int = 0,
        widget_button: int = 0,
        selection_rgba: Tuple[int, int, int, int] = (255, 0, 0, 255),
        parent: QtWidgets.QWidget = None,
    ) -> None:
        super().__init__(
            move_button=move_button,
            widget_button=widget_button,
            selection_rgba=selection_rgba,
            parent=parent,
        )
        self.viewoptions = viewoptions

        self.cax: Axes = None
        self.label: AnchoredText = None
        self.scalebar: MetricSizeBar = None

    def drawFigure(self) -> None:
        super().drawFigure()
        if self.viewoptions.canvas.colorbar:
            div = make_axes_locatable(self.ax)
            self.cax = div.append_axes(
                self.viewoptions.canvas.colorbarpos, size=0.1, pad=0.05
            )

    def drawColorbar(self, label: str) -> None:
        self.cax.clear()
        if self.viewoptions.canvas.colorbarpos in ["right", "left"]:
            orientation = "vertical"
        else:
            orientation = "horizontal"

        self.figure.colorbar(
            self.image,
            label=label,
            ax=self.ax,
            cax=self.cax,
            orientation=orientation,
            ticks=MaxNLocator(nbins=6),
        )
        self.cax.tick_params(labelsize=self.viewoptions.font.size)

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

        # Plot the image
        self.image = self.ax.imshow(
            data,
            cmap=self.viewoptions.image.cmap,
            interpolation=self.viewoptions.image.interpolation,
            alpha=self.viewoptions.image.alpha,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            origin="upper",
        )

    def drawLabel(self, text: str) -> None:
        if self.label is not None:
            self.label.remove()

        self.label = AnchoredText(
            text,
            "upper left",
            pad=0.5,
            borderpad=0,
            frameon=False,
            prop=self.viewoptions.font.props(),
        )
        self.ax.add_artist(self.label)

    def drawScalebar(self) -> None:
        if self.scalebar is not None:
            self.scalebar.remove()

        self.scalebar = MetricSizeBar(
            self.ax,
            loc="upper right",
            color=self.viewoptions.font.color,
            edgecolor="black",
            edgewidth=1.5,
            font_properties=self.viewoptions.font.mpl_props(),
        )
        self.ax.add_artist(self.scalebar)

    def drawLaser(self, laser: Laser, name: str, layer: int = None) -> None:
        # Get the trimmed and calibrated data
        kwargs = {"calibrate": self.viewoptions.calibrate, "layer": layer, "flat": True}

        data = laser.get(name, **kwargs)
        unit = str(laser.calibration[name].unit) if self.viewoptions.calibrate else ""

        # Get extent
        extent = laser.config.data_extent(data.shape, layer=layer)
        # Only change the view if new or the laser extent has changed (i.e. conf edit)
        if self.extent != extent:
            self.view_limits = self.extentForAspect(extent)

        # If data is empty create a dummy data
        if data is None or data.size == 0:
            data = np.array([[0]], dtype=np.float64)

        self.drawData(data, extent, name)

        if self.viewoptions.canvas.colorbar:
            self.drawColorbar(unit)

        if self.viewoptions.canvas.label:
            self.drawLabel(name)
        elif self.label is not None:
            self.label.remove()
            self.label = None

        if self.viewoptions.canvas.scalebar:
            self.drawScalebar()
        elif self.scalebar is not None:
            self.scalebar.remove()
            self.scalebar = None

        # Selection drawing all handled in SelectableImageCanvas
        self.drawSelection()
        self.draw_idle()
