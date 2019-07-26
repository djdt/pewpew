import numpy as np
from matplotlib.image import AxesImage
from matplotlib.transforms import Bbox, BboxTransform

from typing import Tuple


def image_extent_to_data(image: AxesImage) -> BboxTransform:
    x0, x1, y0, y1 = image.get_extent()
    ny, nx = image.get_array().shape[:2]
    if image.origin == "upper":
        y0, y1 = y1, y0
    return BboxTransform(
        boxin=Bbox([[x0, y0], [x1, y1]]), boxout=Bbox([[0, 0], [nx, ny]])
    )


def mask_to_image(
    mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5
) -> np.ndarray:
    assert mask.dtype == np.bool
    image = np.zeros((*mask.shape[:2], 4), dtype=np.uint8)
    image[mask] = [*color, int(alpha * 255)]
    return image
