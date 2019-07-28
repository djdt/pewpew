import numpy as np
from matplotlib.widgets import _SelectorWidget, RectangleSelector, LassoSelector
from matplotlib.backend_bases import MouseEvent
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxTransform
from matplotlib.image import AxesImage


def image_extent_to_data(image: AxesImage) -> BboxTransform:
    x0, x1, y0, y1 = image.get_extent()
    ny, nx = image.get_array().shape[:2]
    if image.origin == "upper":
        y0, y1 = y1, y0
    return BboxTransform(
        boxin=Bbox([[x0, y0], [x1, y1]]), boxout=Bbox([[0, 0], [nx, ny]])
    )


class _ImageSelectionWidget(object):
    state_modifier_keys = dict(move="", clear="", add="shift", subtract="control")

    def __init__(self, selector: _SelectorWidget, image: AxesImage, rgba: np.ndarray):
        self.selector = selector
        self.selector.state_modifier_keys = _ImageSelectionWidget.state_modifier_keys

        self.image = image
        assert len(rgba) == 4
        self.rgba = rgba

        self.mask: np.ndarray = np.zeros(
            self.image.get_array().shape[:2], dtype=np.bool
        )
        self.mask_image = AxesImage(
            image.axes,
            extent=image.get_extent(),
            interpolation="none",
            origin=image.origin,
            visible=False,
            animated=self.selector.useblit,
        )
        self.update_mask_image()

        self.selector.ax.add_image(self.mask_image)
        self.selector.artists.append(self.mask_image)

    def set_color(self, r: int, g: int, b: int) -> None:
        self.rgba[:3] = [r, g, b]

    def set_alpha(self, alpha: float) -> None:
        self.rgba[3] = np.uint8(alpha * 255)

    def update_mask_image(self) -> None:
        assert self.mask.dtype == np.bool
        image = np.zeros((*self.mask.shape[:2], 4), dtype=np.uint8)
        image[self.mask] = self.rgba
        self.mask_image.set_data(image)
        # TODO possible optimisation
        if np.all(self.mask == 0):
            self.mask_image.set_visible(False)
        else:
            self.mask_image.set_visible(True)


class LassoImageSelectionWidget(_ImageSelectionWidget):
    def __init__(
        self,
        image: AxesImage,
        rgba: np.ndarray = (255, 255, 255, 128),
        useblit: bool = False,
        button: int = 1,
        lineprops: dict = None,
    ):
        selector = LassoSelector(
            image.axes,
            self.onselect,
            useblit=useblit,
            button=button,
            lineprops=lineprops,
        )
        super().__init__(selector, image, rgba)

    def onselect(self, vertices: np.ndarray) -> None:
        data = self.image.get_array()
        x0, x1, y0, y1 = self.image.get_extent()
        # Transform verticies into data coords
        transform = image_extent_to_data(self.image)
        vertices = transform.transform(vertices)
        vx = np.array([np.min(vertices[:, 0]), 1 + np.max(vertices[:, 0])], dtype=int)
        vy = np.array([np.min(vertices[:, 1]), 1 + np.max(vertices[:, 1])], dtype=int)
        # Generate point mesh
        x = np.linspace(vx[0] + 0.5, vx[1] + 0.5, vx[1] - vx[0], endpoint=False)
        y = np.linspace(vy[0] + 0.5, vy[1] + 0.5, vy[1] - vy[0], endpoint=False)
        X, Y = np.meshgrid(x, y)
        pix = np.vstack((X.flatten(), Y.flatten())).T

        path = Path(vertices)
        ind = path.contains_points(pix)

        # Refresh the mask if not adding / subtracting to it
        if not any(state in self.selector.state for state in ["add", "subtract"]):
            self.mask = np.zeros(data.shape[:2], dtype=bool)
        # Update the mask
        self.mask[vy[0] : vy[1], vx[0] : vx[1]].flat[ind] = (
            False if "subtract" in self.selector.state else True
        )
        # Update the image
        self.update_mask_image()
        self.selector.line.set_visible(False)
        self.selector.update()


class RectangleImageSelectionWidget(_ImageSelectionWidget):
    def __init__(
        self,
        image: AxesImage,
        rgba: np.ndarray = (255, 255, 255, 128),
        useblit: bool = False,
        button: int = 1,
        rectprops: dict = None,
    ):
        selector = RectangleSelector(
            image.axes,
            self.onselect,
            useblit=useblit,
            button=button,
            rectprops=rectprops,
            drawtype="box",
            interactive=False,
        )
        super().__init__(selector, image, rgba)

    def onselect(self, press: MouseEvent, release: MouseEvent) -> None:
        data = self.image.get_array()
        x0, x1, y0, y1 = self.image.get_extent()
        # Transform verticies into data coords
        transform = image_extent_to_data(self.image)
        vx0, vy0 = transform.transform([press.xdata, release.ydata])
        vx1, vy1 = transform.transform([release.xdata, press.ydata])
        vx = np.array([vx0, 1 + vx1], dtype=int)
        vy = np.array([vy0, 1 + vy1], dtype=int)

        # Refresh the mask if not adding / subtracting to it
        if not any(state in self.selector.state for state in ["add", "subtract"]):
            self.mask = np.zeros(data.shape[:2], dtype=bool)
        # Update the mask
        self.mask[vy[0] : vy[1], vx[0] : vx[1]] = (
            False if "subtract" in self.selector.state else True
        )
        # Update the image
        self.update_mask_image()
        self.selector.update()
