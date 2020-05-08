import numpy as np

from typing import Tuple


def greyscale_to_rgb(array: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Convert a gret scale image to a single color rgb image.

    The image is clipped to 0.0 to 1.0.

    Args:
        array: Image
        rgb: 3 or 4 color array (rgb / rgba)
"""
    array = np.clip(array, 0.0, 1.0)
    return array[..., None] * np.array(rgb, dtype=float)


def normalise(x: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Normalise an array.

    Args:
        x: Array
        vmin: New minimum
        vmax: New maxmimum
"""
    x = (x - x.min()) / x.max()
    x *= vmax - vmin
    x += vmin
    return x


def otsu(x: np.ndarray) -> float:
    """Calculates the otsu threshold of the input array.
    https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/thresholding.py
"""
    hist, bin_edges = np.histogram(x, bins=256)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]

    u1 = np.cumsum(hist * bin_centers) / w1
    u2 = (np.cumsum((hist * bin_centers)[::-1]) / w2[::-1])[::-1]

    i = np.argmax(w1[:-1] * w2[1:] * (u1[:-1] - u2[1:]) ** 2)
    return bin_centers[i]


def view_as_blocks(x: np.ndarray, block: Tuple[int, int], step: Tuple[int, int] = None) -> np.ndarray:
    """Create block sized views into a array, offset by step amount.
    https://github.com/scikit-image/scikit-image/blob/master/skimage/util/shape.py

    Args:
        x: The array.
        block: The size of the view.
        step: Size of step, defaults to block.

    Returns:
        An array of views.
    """
    assert len(block) == x.ndim
    if step is None:
        step = block
    x = np.ascontiguousarray(x)
    shape = tuple((np.array(x.shape) - block) // np.array(step) + 1) + block
    strides = tuple(x.strides * np.array(step)) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def shuffle_blocks(
    x: np.ndarray,
    block: Tuple[int, int],
    mask: np.ndarray = None,
    mask_all: bool = True,
) -> np.ndarray:
    """Shuffle a 2d array as tiles of a certain size.
    If a mask is passed then only the region within the mask is shuffled.
    If mask_all is True then only entirely masked blocks are shuffled otherwise
    even partially masked blocks will be shuffled.

    Args:
        x: Input array.
        block: Size of the tiles.
        mask: Optional mask data.
        mask_all: Only shuffle entirely masked blocks.
"""
    # Pad the array to fit the blocksize
    px, py = block[0] - x.shape[0] % block[0], block[1] - x.shape[1] % block[1]
    blocks = view_as_blocks(np.pad(x, ((0, px), (0, py)), mode="edge"), block)
    shape = blocks.shape
    blocks = blocks.reshape(-1, *block)

    if mask is not None:
        mask = view_as_blocks(
            np.pad(mask, ((0, px), (0, py)), mode="edge"), block
        ).reshape(-1, *block)
        # Shuffle blocks with all or some masked pixels
        mask = np.all(mask, axis=(1, 2)) if mask_all else np.any(mask, axis=(1, 2))

        blocks[mask] = np.random.permutation(blocks[mask])
    else:  # Just shuffle inplace
        np.random.shuffle(blocks)

    # Reform the image and then trim off excess
    return np.hstack(np.hstack(blocks.reshape(shape)))[: x.shape[0], : x.shape[1]]
