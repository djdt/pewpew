
import numpy as np
from PySide6 import QtCore, QtGui


class GraphicsOptions(QtCore.QObject):
    """This object stores information used by pewpew to draw images.

    Parameters:
        colortable: colortable to draw with, see colortables
        colorrange_default: default colorrange to use
        smoothing: whether to smooth images
        calibrate: whether to calibrate images
        font: font to use
        font_color: color fo fonts
        units: unit of image
    """

    colortables = {
        "balance": "Blue to red diverging colormap from cmocean.",
        "cividis": "Perceptually uniform colormap.",
        "cubehelix": "Perceptually uniform colormap, Green, D. A. 2011.",
        "curl": "Green to red diverging colormap from cmocean.",
        "grey": "Simple black to white gradient.",
        "magma": "Perceptually uniform colormap.",
        "inferno": "Perceptually uniform colormap.",
        "turbo": "Google's improved version of rainbow colormap jet.",
        "viridis": "Perceptually uniform colormap.",
    }

    fontOptionsChanged = QtCore.Signal()
    imageOptionsChanged = QtCore.Signal()
    visiblityOptionsChanged = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.color_ranges: dict[str, tuple[float | str, float | str]] = {}
        self.color_range_default: tuple[float | str, float | str] = (0.0, "99%")

        self.scalebar = True
        self.highlight_focus = True
        self.smoothing = False

        self.font = QtGui.QFont("sans", 16)
        self.font_color = QtGui.QColor(255, 255, 255)

        self.calibrate = True
        self.units = "μm"

    @property
    def colortable(self) -> str:
        return QtCore.QSettings().value("Options/Colortable", "viridis")

    @colortable.setter
    def colortable(self, colortable: str) -> None:
        assert colortable in GraphicsOptions.colortables.keys()
        QtCore.QSettings().setValue("Options/Colortable", colortable)

    @property
    def nan_color(self) -> QtGui.QColor:
        return QtCore.QSettings().value(
            "Options/NanColor", QtGui.QColor(0, 0, 0), type=QtGui.QColor
        )

    @nan_color.setter
    def nan_color(self, color: QtGui.QColor) -> None:
        return QtCore.QSettings().setValue("Options/NanColor", color)

    def setFont(self, font: QtGui.QFont) -> None:
        self.font = font
        self.fontOptionsChanged.emit()

    def setFontSize(self, size: int) -> None:
        self.font.setPointSize(size)
        self.fontOptionsChanged.emit()

    def setHighlightFocus(self, hightlight: bool) -> None:
        self.highlight_focus = hightlight
        self.imageOptionsChanged.emit()

    def setScalebarVisible(self, visible: bool) -> None:
        self.scalebar = visible
        self.visiblityOptionsChanged.emit()

    def setSmoothing(self, smooth: bool) -> None:
        self.smoothing = smooth
        self.imageOptionsChanged.emit()

    def get_color_range_as_float(
        self, name: str, data: np.ndarray
    ) -> tuple[float, float]:
        """Get colorrange for 'name' or a default.

        Converts percentile ranges to float values.
        """
        vmin, vmax = self.color_ranges.get(name, self.color_range_default)
        if data.dtype == bool:
            return 0, 1

        if isinstance(vmin, str):
            vmin = np.nanpercentile(data, float(vmin.rstrip("%")))
        if isinstance(vmax, str):
            vmax = np.nanpercentile(data, float(vmax.rstrip("%")))
        return vmin, vmax  # type: ignore

    def get_color_range_as_percentile(
        self, name: str, data: np.ndarray
    ) -> tuple[float, float]:
        """Get colorrange for 'name' or a default.

        Converts float values to percentile ranges.
        """
        vmin, vmax = self.color_ranges.get(name, self.color_range_default)
        if isinstance(vmin, str):
            vmin = float(vmin.rstrip("%"))
        else:
            vmin = np.count_nonzero(data < vmin) / data.size * 100
        if isinstance(vmax, str):
            vmax = float(vmax.rstrip("%"))
        else:
            vmin = np.count_nonzero(data < vmax) / data.size * 100
        return vmin, vmax  # type: ignore

    # def set_colorrange(
    #     self, name: str, colorrange: tuple[float | str, Union[float, str]]
    # ) -> None:
    #     """Set colorrange for 'name'."""
    #     self._colorranges[name] = colorrange
