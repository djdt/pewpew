import numpy as np
from matplotlib.widgets import _SelectorWidget
from matplotlib.lines import Line2D
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


class _ImageSelectionWidget(_SelectorWidget):
    STATE_MODIFIER_KEYS = dict(move="", clear="escape", add="shift", subtract="control")

    def __init__(
        self, image: AxesImage, mask_rgba: np.ndarray, useblit=True, button: int = None
    ):
        self.image = image
        assert len(mask_rgba) == 4
        self.rgba = mask_rgba

        super().__init__(
            image.axes,
            None,
            useblit=useblit,
            button=button,
            state_modifier_keys=self.STATE_MODIFIER_KEYS,
        )
        self.verts: np.ndarray = None

        self.mask: np.ndarray = np.zeros(
            self.image.get_array().shape[:2], dtype=np.bool
        )

        self.mask_image = AxesImage(
            image.axes,
            extent=image.get_extent(),
            transform=image.get_transform(),
            interpolation="none",
            origin=image.origin,
            visible=False,
            animated=useblit,
        )
        self.mask_image.set_data(np.zeros((*self.mask.shape, 4), dtype=np.uint8))

        self.ax.add_image(self.mask_image)
        self.artists = [self.mask_image]

    def _press(self, event: MouseEvent) -> None:
        self.verts = [self._get_data(event)]
        self.line.set_visible(True)

    def _release(self, event: MouseEvent) -> None:
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            self.update_mask(self.verts)

        self.line.set_data([[], []])
        self.line.set_visible(False)
        self.verts = None
        self.update()

    def update_mask(self, vertices: np.ndarray) -> None:
        data = self.image.get_array()
        x0, x1, y0, y1 = self.image.get_extent()

        # Transform verticies into image data coords
        transform = self.ax.transData + self.image.get_transform().inverted()
        vertices = transform.transform(vertices)

        # Transform verticies into array coords
        transform = image_extent_to_data(self.image)
        vertices = transform.transform(vertices)
        vx = np.array([np.min(vertices[:, 0]), 1 + np.max(vertices[:, 0])], dtype=int)
        vy = np.array([np.min(vertices[:, 1]), 1 + np.max(vertices[:, 1])], dtype=int)

        # Bound to array size
        vx[0] = max(0, vx[0])
        vx[1] = min(data.shape[1], vx[1])
        vy[0] = max(0, vy[0])
        vy[1] = min(data.shape[0], vy[1])

        # If somehow the data is malformed then return
        if vx[1] - vx[0] < 1 or vy[1] - vy[0] < 1:
            return

        # Generate point mesh
        x = np.linspace(vx[0] + 0.5, vx[1] + 0.5, vx[1] - vx[0], endpoint=False)
        y = np.linspace(vy[0] + 0.5, vy[1] + 0.5, vy[1] - vy[0], endpoint=False)
        X, Y = np.meshgrid(x, y)
        pix = np.vstack((X.flatten(), Y.flatten())).T

        path = Path(vertices)
        ind = path.contains_points(pix)

        # Refresh the mask if not adding / subtracting to it
        if not any(state in self.state for state in ["add", "subtract"]):
            self.mask = np.zeros(data.shape[:2], dtype=bool)
        # Update the mask
        self.mask[vy[0] : vy[1], vx[0] : vx[1]].flat[ind] = (
            False if "subtract" in self.state else True
        )

        self._update_mask_image()

    def _update_mask_image(self) -> None:
        assert self.mask.dtype == np.bool

        image = np.zeros((*self.mask.shape[:2], 4), dtype=np.uint8)
        image[self.mask] = self.rgba
        self.mask_image.set_data(image)

        if np.all(self.mask == 0):
            self.mask_image.set_visible(False)
        else:
            self.mask_image.set_visible(True)


class LassoImageSelectionWidget(_ImageSelectionWidget):
    def __init__(
        self,
        image: AxesImage,
        mask_rgba: np.ndarray = (255, 255, 255, 128),
        useblit: bool = True,
        button: int = 1,
        lineprops: dict = None,
    ):
        super().__init__(image, mask_rgba, useblit=useblit, button=button)

        if lineprops is None:
            lineprops = dict()
        if useblit:
            lineprops["animated"] = True

        self.line = Line2D([], [], **lineprops)
        self.line.set_visible(False)
        self.ax.add_line(self.line)
        self.artists.append(self.line)

    def _onmove(self, event: MouseEvent) -> None:
        if self.verts is None:
            return
        self.verts.append(self._get_data(event))
        self.line.set_data(list(zip(*self.verts)))
        self.update()


class RectangleImageSelectionWidget(_ImageSelectionWidget):
    def __init__(
        self,
        image: AxesImage,
        mask_rgba: np.ndarray = (255, 255, 255, 128),
        useblit: bool = True,
        button: int = 1,
        lineprops: dict = None,
    ):
        super().__init__(image, mask_rgba, useblit=useblit, button=button)

        if lineprops is None:
            lineprops = dict()
        if useblit:
            lineprops["animated"] = True

        self.line = Line2D([], [], **lineprops)
        self.line.set_visible(False)
        self.ax.add_line(self.line)
        self.artists.append(self.line)

    def _onmove(self, event: MouseEvent) -> None:
        if self.eventpress is None:
            return
        x0, y0 = self._get_data(self.eventpress)
        x1, y1 = self._get_data(event)
        self.line.set_data([[x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0]])
        self.update()

    def _release(self, event: MouseEvent) -> None:
        if self.eventpress is not None:
            x0, y0 = self._get_data(self.eventpress)
            x1, y1 = self._get_data(event)
            self.update_mask([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])

        self.line.set_data([[], []])
        self.line.set_visible(False)
        self.update()
 
