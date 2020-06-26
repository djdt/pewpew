import numpy as np

from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D, Bbox, BboxTransform, Transform
from matplotlib.text import Text
from matplotlib.patheffects import withStroke

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from typing import Dict, List, Tuple


class LabeledLine2D(Line2D):
    def __init__(self, *args, **kwargs) -> None:
        self.label_at = kwargs.pop("label_at", 0)
        self.label_offset = kwargs.pop("label_offset", (0, 0))
        self.text = Text(0, 0, "")
        super().__init__(*args, **kwargs)
        self.text.set_text(self.get_label())

    def set_axes(self, axes: Axes) -> None:
        self.text.set_axes(axes)
        super().set_axes(axes)

    def set_figure(self, figure: Figure) -> None:
        self.text.set_figure(figure)
        super().set_figure(figure)

    def set_transform(self, transform: Transform) -> None:
        text_transform = transform + Affine2D().translate(*self.label_offset)
        self.text.set_transform(text_transform)
        super().set_transform(transform)

    def set_xdata(self, x: np.ndarray) -> None:
        self.text.set_x(x[self.label_at])
        super().set_xdata(x)

    def set_ydata(self, y: np.ndarray) -> None:
        self.text.set_y(y[self.label_at])
        super().set_ydata(y)

    def draw(self, renderer: RendererBase) -> None:
        super().draw(renderer)
        self.text.draw(renderer)


class MetricSizeBar(AnchoredSizeBar):
    allowed_lengths = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    units = {
        "pm": 1e-12,
        "nm": 1e-9,
        "μm": 1e-6,
        "mm": 1e-3,
        "cm": 1e-2,
        "m": 1.0,
        "km": 1e3,
    }

    def __init__(
        self,
        axes: Axes,
        axes_unit: str = "μm",
        loc: str = "upper right",
        bar_height_fraction: float = 0.01,
        color: str = "white",
        edgecolor: str = "black",
        edgewidth: float = 1.5,
        font_properties: FontProperties = None,
    ):
        self.min_length = list(self.units.values())[0] * self.allowed_lengths[0]
        self.max_length = list(self.units.values())[-1] * self.allowed_lengths[-1]
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
        # Give a black outline to the bar and text
        if edgecolor is not None:
            rect = self.size_bar.get_children()[0]
            rect.set_edgecolor(None)
            rect.set_path_effects(
                [withStroke(linewidth=edgewidth, foreground=edgecolor)]
            )
            self.txt_label._text.set_path_effects(
                [withStroke(linewidth=edgewidth, foreground=edgecolor)]
            )

    def get_bar_height(self) -> float:
        return abs(self.axes.get_ylim()[1] - self.axes.get_ylim()[0]) * self.bar_height

    def get_bar_width_and_unit(self) -> Tuple[float, str]:
        desired = abs(self.axes.get_xlim()[1] - self.axes.get_xlim()[0]) * 0.2
        base = desired * self.units[self.unit]

        units = list(self.units.keys())
        factors = list(self.units.values())
        idx = np.max(np.searchsorted(factors, base) - 1, 0)

        new = self.allowed_lengths[
            np.searchsorted(self.allowed_lengths, base / factors[idx]) - 1
        ]
        new_unit = units[idx]

        return new * factors[idx] / self.units[self.unit], new_unit

    def draw(self, renderer: RendererBase, *args, **kwargs) -> None:
        width, unit = self.get_bar_width_and_unit()
        rect = self.size_bar.get_children()[0]
        if width < self.min_length or width > self.max_length:
            rect.set_width(0)
            rect.set_height(0)
            self.txt_label.set_text("No Scale")
        else:
            rect.set_width(width)
            rect.set_height(self.get_bar_height())
            factor = self.units[unit] / self.units[self.unit]
            self.txt_label.set_text(f"{width / factor:.0f} {unit}")
        super().draw(renderer, *args, **kwargs)


def image_extent_to_data(image: AxesImage) -> BboxTransform:
    x0, x1, y0, y1 = image.get_extent()
    ny, nx = image.get_array().shape[:2]
    if image.origin == "upper":
        y0, y1 = y1, y0
    return BboxTransform(
        boxin=Bbox([[x0, y0], [x1, y1]]), boxout=Bbox([[0, 0], [nx, ny]])
    )
