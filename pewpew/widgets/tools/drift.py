import numpy as np

from PySide2 import QtCore, QtWidgets

from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.backend_bases import PickEvent, MouseEvent
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
from matplotlib.transforms import blended_transform_factory

from mpl_toolkits.axes_grid1 import make_axes_locatable

from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import InteractiveImageCanvas
from pewpew.widgets.laser import LaserWidget

from .tool import ToolWidget

from typing import List, Tuple


class DriftCanvas(InteractiveImageCanvas):
    guidesChanged = QtCore.Signal()

    def __init__(self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None):
        super().__init__(widget_button=1, state=(), parent=parent)
        self.viewoptions = viewoptions

        self.connect_event("draw_event", self.update_background)
        self.connect_event("resize_event", self._resize)
        self.connect_event("pick_event", self._pick)

        self.guides_need_draw = True
        self.draw_edge_guides = False

        self.background = None

        self.guides_x: List[Line2D] = []
        self.guides_y: List[Line2D] = []
        self.guides_rect: Rectangle = None

        self.picked_artist: Artist = None

        self.cax: Axes = None
        self.cbackground = None

        self.drawFigure()

    def drawFigure(self) -> None:
        # TODO: whitespace?
        super().drawFigure()
        div = make_axes_locatable(self.ax)
        self.cax = div.append_axes("bottom", size=1.0, pad=0.1, xmargin=0, ymargin=0)
        self.cax.get_xaxis().set_visible(False)
        self.cax.get_yaxis().set_visible(False)

    def _resize(self, event) -> None:
        self.guides_need_draw = True

    def _pick(self, event: PickEvent) -> None:
        if self.ignore_event(event.mouseevent):
            return
        if event.mouseevent.button == self.widget_button:
            self.picked_artist = event.artist
        else:
            self.picked_artist = None

    def update_background(self, event) -> None:
        self.background = self.copy_from_bbox(self.ax.bbox)
        if self.guides_need_draw:
            self.blitGuides()

    def move(self, event: MouseEvent) -> None:
        if self.picked_artist is not None:
            if self.picked_artist in self.guides_x:
                self.picked_artist.set_xdata([event.xdata, event.xdata])
            elif self.picked_artist in self.guides_y:
                self.picked_artist.set_ydata([event.ydata, event.ydata])
            self.blitGuides()

    def release(self, event: MouseEvent) -> None:
        if self.picked_artist is not None:
            x = self.picked_artist.get_xdata()[0]
            y = self.picked_artist.get_ydata()[0]
            shape = self.image.get_array().shape
            x0, x1, y0, y1 = self.extent
            # Snap to pixels
            pxy = (x1 - x0, y1 - y0) / np.array((shape[1], shape[0]))
            x, y = pxy * np.round((x, y) / pxy)

            if self.picked_artist in self.guides_x:
                self.picked_artist.set_xdata([x, x])
            elif self.picked_artist in self.guides_y:
                self.picked_artist.set_ydata([y, y])

            self.blitGuides()
            self.guidesChanged.emit()
            self.picked_artist = None

    def blitGuides(self) -> None:
        if self.background is not None:
            self.restore_region(self.background)

        if self.draw_edge_guides:
            for a in self.guides_y:
                self.ax.draw_artist(a)
        for a in self.guides_x:
            self.ax.draw_artist(a)

        # self.ax.draw_artist(self.guides_rect)

        self.blit(self.ax.bbox)
        self.guides_need_draw = False

    def drawData(
        self,
        data: np.ndarray,
        extent: Tuple[float, float, float, float],
        isotope: str = None,
    ) -> None:
        if self.image is not None:  # pragma: no cover
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
            origin="lower",
        )

    def drawDrift(self, x: np.ndarray, y: np.ndarray, fit: np.ndarray = None):
        self.cax.clear()

        if fit is not None:
            self.cax.plot(x, y, lw=0.75, color="black")
        else:  # So we dont have to plot twice if no fit needed
            fit = y

        self.cax.plot(x, fit, color="red")

        self.draw_idle()
        self.guides_need_draw = True

    def drawGuides(self, xs: Tuple[float, float], ys: Tuple[float, float]) -> None:
        for line in self.guides_x:
            line.remove()
        self.guides_x = []
        for line in self.guides_y:
            line.remove()
        self.guides_y = []

        for x in xs:
            line = Line2D(
                (x, x),
                (0.0, 1.0),
                transform=blended_transform_factory(
                    self.ax.transData, self.ax.transAxes
                ),
                color="white",
                linestyle="-",
                path_effects=[withStroke(linewidth=2.0, foreground="black")],
                linewidth=1.0,
                picker=True,
                pickradius=5,
                animated=True,
            )
            self.guides_x.append(line)
            self.ax.add_artist(line)

        for y in ys:
            line = Line2D(
                (0.0, 1.0),
                (y, y),
                transform=blended_transform_factory(
                    self.ax.transAxes, self.ax.transData
                ),
                color="white",
                linestyle="--",
                path_effects=[withStroke(linewidth=2.0, foreground="black")],
                linewidth=1.0,
                picker=True,
                pickradius=5,
                animated=True,
            )
            self.guides_y.append(line)
            self.ax.add_artist(line)

        # rx = min(line.get_xdata() for line in self.guides_x)
        # ry = min(line.get_xdata() for line in self.guides_x)
        # self.guides_rect = Rectangle((np.amin(xs), 0.0, ))

    def getCurrentXTrim(self) -> Tuple[int, int]:
        shape = self.image.get_array().shape
        x0, x1, y0, y1 = self.extent
        px = (x1 - x0) / shape[1]  # Data coords
        trim = np.array(
            [guide.get_xdata()[0] / px for guide in self.guides_x], dtype=int
        )

        return np.min(trim), np.max(trim)

    def getCurrentYTrim(self) -> Tuple[int, int]:
        shape = self.image.get_array().shape
        x0, x1, y0, y1 = self.extent
        py = (y1 - y0) / shape[0]  # Data coords
        trim = np.array(
            [guide.get_ydata()[0] / py for guide in self.guides_y], dtype=int
        )

        return np.min(trim), np.max(trim)

    def getDriftData(self) -> np.ndarray:
        start, end = self.getCurrentXTrim()
        data = self.image.get_array()[:, start:end].copy()
        if self.draw_edge_guides:
            start, end = self.getCurrentYTrim()
            data[start:end] = np.nan
        return data

    def setEdgeGuidesVisible(self, visible: bool) -> None:
        self.draw_edge_guides = visible
        self.blitGuides()


class DriftTool(ToolWidget):
    normalise_methods = ["Maximum", "Minimum"]

    def __init__(self, widget: LaserWidget):
        super().__init__(widget, apply_all=False)

        self.drift: np.ndarray = None

        self.canvas = DriftCanvas(self.viewspace.options, parent=self)
        self.canvas.state.discard("move")  # Prevent moving
        self.canvas.state.discard("scroll")  # Prevent scroll zoom
        self.canvas.guidesChanged.connect(self.updateDrift)

        # self.drift_plot = DriftPlotCanvas(parent=self)
        # self.drift_plot.drawFigure()

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.activated.connect(self.refresh)

        self.spinbox_degree = QtWidgets.QSpinBox()
        self.spinbox_degree.setRange(0, 9)
        self.spinbox_degree.setValue(3)
        self.spinbox_degree.valueChanged.connect(self.updateDrift)
        self.spinbox_degree.setToolTip(
            "Degree of polynomial used to fit the drift,\nuse 0 for the raw data."
        )

        self.combo_normalise = QtWidgets.QComboBox()
        self.combo_normalise.addItems(DriftTool.normalise_methods)
        self.combo_normalise.setCurrentText("Minimum")
        self.combo_normalise.activated.connect(self.updateNormalise)

        self.lineedit_normalise = QtWidgets.QLineEdit()
        self.lineedit_normalise.setEnabled(False)

        self.check_trim = QtWidgets.QCheckBox("Show drift trim controls.")
        self.check_trim.toggled.connect(self.canvas.setEdgeGuidesVisible)
        self.check_trim.toggled.connect(self.updateDrift)

        self.check_apply_all = QtWidgets.QCheckBox("Apply to all elements.")

        layout_canvas = QtWidgets.QVBoxLayout()
        layout_canvas.addWidget(self.canvas)
        # layout_canvas.addWidget(self.drift_plot)
        layout_canvas.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignRight)
        self.box_canvas.setLayout(layout_canvas)

        layout_norm = QtWidgets.QVBoxLayout()
        layout_norm.addWidget(self.combo_normalise)
        layout_norm.addWidget(self.lineedit_normalise)

        layout_controls = QtWidgets.QFormLayout()
        layout_controls.addRow("Degree of fit:", self.spinbox_degree)
        layout_controls.addRow("Normalise to:", layout_norm)
        layout_controls.addRow(self.check_trim)
        layout_controls.addRow(self.check_apply_all)
        self.box_controls.setLayout(layout_controls)

        self.initialise()

    def apply(self) -> None:
        if self.combo_normalise.currentText() == "Maximum":
            value = np.amax(self.drift)
        elif self.combo_normalise.currentText() == "Minimum":
            value = np.amin(self.drift)

        if self.check_apply_all.isChecked():
            names = self.widget.laser.isotopes
        else:
            names = [self.combo_isotope.currentText()]

        for name in names:
            transpose = self.widget.laser.data[name].T
            transpose /= self.drift / value

        self.refresh()

    def initialise(self) -> None:
        isotopes = self.widget.laser.isotopes
        self.combo_isotope.clear()
        self.combo_isotope.addItems(isotopes)

        self.refresh()

    def isComplete(self) -> bool:
        return self.drift is not None

    def updateDrift(self) -> None:
        ys = self.canvas.getDriftData()
        ys = np.nanmean(ys, axis=1)

        xs = np.arange(ys.size)

        nans = np.isnan(ys)

        deg = self.spinbox_degree.value()
        if deg == 0:
            self.drift = ys
            self.canvas.drawDrift(xs, ys, None)
        else:
            coef = np.polynomial.polynomial.polyfit(xs[~nans], ys[~nans], deg)
            self.drift = np.polynomial.polynomial.polyval(xs, coef)
            self.canvas.drawDrift(xs, ys, self.drift)

    def updateNormalise(self) -> None:
        if self.combo_normalise.currentText() == "Maximum":
            value = np.amax(self.drift)
        elif self.combo_normalise.currentText() == "Minimum":
            value = np.amin(self.drift)
        self.lineedit_normalise.setText(f"{value:.8g}")

    def refresh(self) -> None:
        isotope = self.combo_isotope.currentText()

        data = self.widget.laser.get(isotope, calibrate=False, flat=True)
        extent = self.widget.laser.config.data_extent(data.shape)
        self.canvas.drawData(data, extent, isotope=isotope)
        # Update view limits
        self.canvas.view_limits = self.canvas.extentForAspect(extent)
        if len(self.canvas.guides_x) == 0:
            w, h = extent[1] - extent[0], extent[3] - extent[2]
            px, py = w / data.shape[1], h / data.shape[0]
            x0, x1 = px * np.round(np.array([0.01, 0.11]) * w / px)
            y0, y1 = py * np.round(np.array([0.1, 0.9]) * h / py)
            self.canvas.drawGuides((x0, x1), (y0, y1))

        self.canvas.guides_need_draw = True
        self.updateDrift()
        self.updateNormalise()
        self.canvas.draw()
