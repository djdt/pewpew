import numpy as np

from PySide2 import QtCore, QtWidgets

from matplotlib.artist import Artist
from matplotlib.backend_bases import PickEvent, MouseEvent
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.transforms import blended_transform_factory

from pewpew.lib.viewoptions import ViewOptions
from pewpew.widgets.canvases import BasicCanvas, InteractiveImageCanvas
from pewpew.widgets.laser import LaserWidget

from .tool import ToolWidget

from typing import List, Tuple


class DriftPlotCanvas(BasicCanvas):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__((1.0, 0.1), parent=parent)

    def drawFigure(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot()
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

    def drawDrift(self, x: np.ndarray, y: np.ndarray, fit: np.ndarray = None):
        self.ax.clear()

        if fit is not None:
            self.ax.plot(x, y, lw=0.75, color="black")
        else:  # So we dont have to plot twixe if no fit needed
            fit = y

        self.ax.plot(x, fit, color="red")
        self.draw_idle()

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(100, 100)


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
        self.drift_guides: List[Line2D] = []
        self.edge_guides: List[Line2D] = []

        self.picked_artist: Artist = None

        self.drawFigure()

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
            if self.picked_artist in self.drift_guides:
                self.picked_artist.set_xdata([event.xdata, event.xdata])
            elif self.picked_artist in self.edge_guides:
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

            if self.picked_artist in self.drift_guides:
                self.picked_artist.set_xdata([x, x])
            elif self.picked_artist in self.edge_guides:
                self.picked_artist.set_ydata([y, y])

            self.blitGuides()
            self.guidesChanged.emit()
            self.picked_artist = None

    def drawData(
        self, data: np.ndarray, extent: Tuple[float, float, float, float]
    ) -> None:
        if self.image is not None:
            self.image.remove()
        self.image = self.ax.imshow(
            data,
            extent=extent,
            cmap=self.viewoptions.image.cmap,
            interpolation=self.viewoptions.image.interpolation,
            alpha=self.viewoptions.image.alpha,
            aspect="equal",
            origin="lower",
        )

    def drawEdgeGuides(self, ypos: Tuple[float, float]) -> None:
        for line in self.edge_guides:
            line.remove()
        self.edge_guides = []

        for y in ypos:
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
            self.edge_guides.append(line)
            self.ax.add_artist(line)

    def drawDriftGuides(self, xpos: Tuple[float, float]) -> None:
        for line in self.drift_guides:
            line.remove()
        self.drift_guides = []

        for x in xpos:
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
            self.drift_guides.append(line)
            self.ax.add_artist(line)

    def blitGuides(self) -> None:
        if self.background is not None:
            self.restore_region(self.background)

        if self.draw_edge_guides:
            for a in self.edge_guides:
                self.ax.draw_artist(a)
        for a in self.drift_guides:
            self.ax.draw_artist(a)

        self.blit()
        self.guides_need_draw = False

    def getCurrentDriftTrim(self) -> Tuple[int, int]:
        shape = self.image.get_array().shape
        x0, x1, y0, y1 = self.extent
        px = (x1 - x0) / shape[1]  # Data coords
        trim = np.array(
            [guide.get_xdata()[0] / px for guide in self.drift_guides], dtype=int
        )

        return np.min(trim), np.max(trim)

    def getCurrentEdgeTrim(self) -> Tuple[int, int]:
        shape = self.image.get_array().shape
        x0, x1, y0, y1 = self.extent
        py = (y1 - y0) / shape[0]  # Data coords
        trim = np.array(
            [guide.get_ydata()[0] / py for guide in self.edge_guides], dtype=int
        )

        return np.min(trim), np.max(trim)

    def getDriftData(self) -> np.ndarray:
        start, end = self.getCurrentDriftTrim()
        data = self.image.get_array()[:, start:end].copy()
        if self.draw_edge_guides:
            start, end = self.getCurrentEdgeTrim()
            data[start:end] = np.nan
        return data

    def setEdgeGuidesVisible(self, visible: bool) -> None:
        self.draw_edge_guides = visible
        self.blitGuides()


class DriftTool(ToolWidget):
    fitting = ["None", "First Degree", "Second Degree", "Third Degree"]
    fittings: dict = {
        "None": {
            "method": None,
            "params": [],
            "desc": [],
        },
        "Linear": {
            "method": None,
            "params": [],
            "desc": [],
        },
        "Polynomial": {
            "method": None,
            "params": [
                ("degree", 2, (2, 10), None),
            ],
            "desc": ["Degree if polynomial fit."],
        },
    }

    def __init__(self, widget: LaserWidget):
        super().__init__(widget, apply_all=True)

        self.canvas = DriftCanvas(self.viewspace.options, parent=self)
        self.canvas.guidesChanged.connect(self.updateDrift)
        self.drift = DriftPlotCanvas(parent=self)
        self.drift.drawFigure()

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.activated.connect(self.refresh)

        self.combo_fitting = QtWidgets.QComboBox()

        self.check_trim = QtWidgets.QCheckBox("Show drift trim controls.")
        self.check_trim.toggled.connect(self.canvas.setEdgeGuidesVisible)
        self.check_trim.toggled.connect(self.updateDrift)

        layout_canvas = QtWidgets.QVBoxLayout()
        layout_canvas.addWidget(self.canvas)
        layout_canvas.addWidget(self.drift)
        self.box_canvas.setLayout(layout_canvas)

        layout_controls = QtWidgets.QFormLayout()
        layout_controls.addWidget(self.check_trim)
        self.box_controls.setLayout(layout_controls)

        self.initialise()

    def initialise(self) -> None:
        isotopes = self.widget.laser.isotopes
        self.combo_isotope.clear()
        self.combo_isotope.addItems(isotopes)

        self.refresh()

    def updateDrift(self) -> None:
        ys = self.canvas.getDriftData()
        ys = np.nanmean(ys, axis=1)

        xs = np.arange(ys.size)

        coef = np.polynomial.polynomial.polyfit(xs, ys, 1)
        fit = np.polynomial.polynomial.polyval(xs, coef)

        self.drift.drawDrift(xs, ys, fit)

    def refresh(self) -> None:
        isotope = self.combo_isotope.currentText()

        data = self.widget.laser.get(isotope, calibrate=False, flat=True)
        extent = self.widget.laser.config.data_extent(data.shape)
        self.canvas.drawData(data, extent)
        # Update view limits
        self.canvas.view_limits = self.canvas.extentForAspect(extent)
        if len(self.canvas.drift_guides) == 0:
            w = extent[1] - extent[0]
            px = w / data.shape[1]
            x0, x1 = px * np.round(np.array([0.01, 0.11]) * w / px)
            self.canvas.drawDriftGuides((x0, x1))
        if len(self.canvas.edge_guides) == 0:
            w = extent[3] - extent[2]
            py = w / data.shape[0]
            y0, y1 = py * np.round(np.array([0.1, 0.9]) * w / py)
            self.canvas.drawEdgeGuides((y0, y1))

        self.canvas.guides_need_draw = True
        self.canvas.draw()
        self.updateDrift()
