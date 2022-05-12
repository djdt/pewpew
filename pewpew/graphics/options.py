from PySide2 import QtCore, QtGui
import numpy as np

from typing import Dict, Tuple, Union


class GraphicsOptions(QtCore.QObject):
    """This object stores information used by pewpew to draw images.

    Parameters:
        items: dict of item visibilities
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
        "curl": "Green to red diverging colormap from cmocean.",
        "grey": "Simple black to white gradient.",
        "magma": "Perceptually uniform colormap.",
        "inferno": "Perceptually uniform colormap.",
        "turbo": "Google's improved version of rainbow colormap jet.",
        "viridis": "Perceptually uniform colormap.",
    }

    optionsChanged = QtCore.Signal()

    def __init__(self) -> None:
        # Todo: maybe alignments here?
        self.overlay_items = {
            "label": True,
            "scalebar": True,
            "colorbar": True,
        }

        self.colortable = "viridis"
        self.color_ranges: Dict[str, Tuple[Union[float, str], Union[float, str]]] = {}
        self.color_range_default = (0.0, "99%")

        self.smoothing = False

        self.font = QtGui.QFont("sans", 16)
        self.font_color = QtGui.QColor(255, 255, 255)

        self.calibrate = True
        self.units = "μm"

    def setFont(self, font: QtGui.QFont) -> None:
        self.font = font
        self.optionsChanged.emit()

    def setFontSize(self, size: int) -> None:
        self.font.setPointSize(size)
        self.optionsChanged.emit()

    def setSmoothing(self, smooth: bool) -> None:
        self.smoothing = smooth
        self.optionsChanged.emit()

    def get_color_range_as_float(
        self, name: str, data: np.ndarray
    ) -> Tuple[float, float]:
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

    # def get_colorrange(self, name: str) -> Tuple[Union[float, str], Union[float, str]]:
    #     """Get colorrange for 'name' or a default."""
    #     return self._colorranges.get(name, self.colorrange_default)

    # def get_colorrange_as_percentile(
    #     self, name: str, data: np.ndarray
    # ) -> Tuple[float, float]:
    #     """Get colorrange for 'name' or a default.

    #     Converts float values to percentile ranges.
    #     """
    #     vmin, vmax = self.get_colorrange(name)
    #     if isinstance(vmin, str):
    #         vmin = float(vmin.rstrip("%"))
    #     else:
    #         vmin = np.count_nonzero(data < vmin) / data.size * 100
    #     if isinstance(vmax, str):
    #         vmax = float(vmax.rstrip("%"))
    #     else:
    #         vmin = np.count_nonzero(data < vmax) / data.size * 100
    #     return vmin, vmax  # type: ignore

    # def set_colorrange(
    #     self, name: str, colorrange: Tuple[Union[float, str], Union[float, str]]
    # ) -> None:
    #     """Set colorrange for 'name'."""
    #     self._colorranges[name] = colorrange
