from matplotlib.colors import Colormap

from pewpew.lib.mplcolors import ppSpectral

from typing import Tuple, Union


class ColorOptions(object):
    COLORMAPS = {
        "Magma": "magma",
        "Viridis": "viridis",
        "PewPew": ppSpectral,
        "Cividis": "cividis",
        "Blue Red": "RdBu_r",
        "Blue Yellow Red": "RdYlBu_r",
        "Grey": "grey",
    }

    def __init__(
        self,
        cmap: Union[str, Colormap] = ppSpectral,
        vmin: float = None,
        vmax: float = None,
    ):
        self.cmap = cmap

        self._range = (vmin, vmax)
        self._ranges = {}

    def get_range(self, isotope: str = None) -> Tuple[float, float]:
        if isotope is None:
            return self._range
        return self._ranges.setdefault(isotope, self._range)

    def set_range(self, range: Tuple[float, float], isotope: str = None) -> None:
        if isotope is None:
            self._range = range
        self._ranges[isotope] = range


class ImageOptions(object):
    INTERPOLATIONS = {"None": "none", "Bilinear": "bilinear", "Bicubic": "bicubic"}

    def __init__(self, interpolation: str = "none", alpha: float = 1.0):
        self.interpolation = interpolation
        self.alpha = alpha


class FontOptions(object):
    def __init__(self, size: int = 12, color: str = "white"):
        self.size = size
        self.color = color


class ViewOptions(object):
    def __init__(self, *args, **kwargs):
        self.colors = ColorOptions()
        # self.filter = {}
        self.image = ImageOptions()
        self.font = FontOptions()

    DEFAULT_VIEW_CONFIG = {
        "cmap": {"type": ppSpectral, "range": (0.0, "99%")},
        "calibrate": True,
        "filtering": {"type": "None", "window": (3, 3), "threshold": 9},
        "interpolation": "None",
        "status_unit": "μm",
        "alpha": 1.0,
        "font": {"size": 12},
    }
