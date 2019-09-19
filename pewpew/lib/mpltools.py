import numpy as np

from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.backend_bases import RendererBase
from matplotlib.image import AxesImage
from matplotlib.transforms import Bbox, BboxTransform

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from typing import Tuple


class MetricSizeBar(AnchoredSizeBar):
    units = {
        "pm": 1e-12,
        "nm": 1e-9,
        "μm": 1e-6,
        "mm": 1e-3,
        "cm": 1e-2,
        "m": 1.0,
        "km": 1e3,
    }
    allowed_lengths = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    def __init__(
        self,
        axes: Axes,
        axes_unit: str = "μm",
        loc: str = "upper right",
        bar_height_fraction: float = 0.01,
        color: str = "white",
        font_properties: FontProperties = None,
    ):
        super().__init__(
            axes.transData,
            0,
            "",
            loc,
            pad=0.5,
            borderpad=0,
            sep=5,
            frameon=False,
            size_vertical=1,
            color=color,
            fontproperties=font_properties,
        )
        self.bar_height = bar_height_fraction
        self.unit = axes_unit

    def get_bar_height(self) -> float:
        return abs(self.axes.get_ylim()[1] - self.axes.get_ylim()[0]) * self.bar_height

    def get_bar_width_and_unit(self) -> Tuple[float, str]:
        desired = abs(self.axes.get_xlim()[1] - self.axes.get_xlim()[0]) * 0.2
        base = desired * self.units[self.unit]

        units = list(self.units.keys())
        factors = list(self.units.values())
        idx = np.searchsorted(factors, base) - 1

        new = base / factors[idx]
        new_unit = units[idx]

        new = self.allowed_lengths[np.searchsorted(self.allowed_lengths, new)]
        return new, new_unit

    def draw(self, renderer: RendererBase, *args, **kwargs) -> None:
        width, unit = self.get_bar_width_and_unit()
        rect = self.size_bar.get_children()[0]
        rect.set_width(width)
        rect.set_height(self.get_bar_height())
        self.txt_label.set_text(f"{width} {unit}")
        super().draw(renderer, *args, **kwargs)


def image_extent_to_data(image: AxesImage) -> BboxTransform:
    x0, x1, y0, y1 = image.get_extent()
    ny, nx = image.get_array().shape[:2]
    if image.origin == "upper":
        y0, y1 = y1, y0
    return BboxTransform(
        boxin=Bbox([[x0, y0], [x1, y1]]), boxout=Bbox([[0, 0], [nx, ny]])
    )
