from pewpew.lib.colormaps import ppSpectral
from typing import Union
from matplotlib.colors import Colormap


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
        self.default_range = (vmin, vmax)

        self.isotope_ranges = {}

    # def set_range(self, range: Tuple[float, float], isotope: str = None) -> bool:
    #     if isotope is None:
    #         self.default_range = range
    #     self.isotope_ranges[isotope] = range

    # def get_range(self, isotope: str = None) -> Tuple[float, float]:
    #     if isotope is None:
    #         return self.default_range
    #     return self.isotope_ranges.setdefault(isotope, self.default_range)


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

        self.status.unit = "um"

    DEFAULT_VIEW_CONFIG = {
        "cmap": {"type": ppSpectral, "range": (0.0, "99%")},
        "calibrate": True,
        "filtering": {"type": "None", "window": (3, 3), "threshold": 9},
        "interpolation": "None",
        "status_unit": "μm",
        "alpha": 1.0,
        "font": {"size": 12},
    }
