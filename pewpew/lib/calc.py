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


def otsu(x: np.ndarray):
    """Calculates the otsu threshold of the input array.

    Implementation from scikit-learn
"""
    hist, bin_edges = np.histogram(x, bins=256)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]

    u1 = np.cumsum(hist * bin_centers) / w1
    u2 = (np.cumsum((hist * bin_centers)[::-1]) / w2[::-1])[::-1]

    i = np.argmax(w1[:-1] * w2[1:] * (u1[:-1] - u2[1:]) ** 2)
    return bin_centers[i]


def colocal_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Returns Pearson's colocalisation coefficient for the two arrays.
    This value for colocalisation between -1 and 1 (colocalised).
"""
    return (np.nanmean(x * y) - (np.nanmean(x) * np.nanmean(y))) / (
        np.nanstd(x) * np.nanstd(y)
    )


def colocal_pearsonr_probablity(
    x: np.ndarray, y: np.ndarray, blocksize: int = 5, n: int = 1000
) -> Tuple[float, float]:
    """Returns Pearson's colocalisation coefficient and the relevant probabilty.
"""
    r = colocal_pearsonr(x, y)
    rs = np.array(
        [
            colocal_pearsonr(x, shuffle_tiles(y, (blocksize, blocksize)))
            for i in range(n)
        ]
    )
    return r, (rs < r).sum() / n


def colocal_costes(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    pearson_r, r_prob = colocal_pearsonr_probablity(x, y, n=200)

    a, b = np.polynomial.Polynomial.fit(x.ravel(), y.ravel(), 1).convert().coef

    # Find the thresholds
    threshold_x = x.max()
    threshold_min = x.min()
    increment = (threshold_x - threshold_min) / 256.0

    r = pearson_r
    while r > 0.0 and threshold_x > threshold_min:
        idx = (x <= threshold_x) | (y <= a * threshold_x + b)
        if np.all(x[idx] == 0) or np.all(y[idx] == 0):
            break
        r = colocal_pearsonr(x[idx], y[idx])
        threshold_x -= increment

    threshold_y = a * threshold_x + b
    idx = (x > threshold_x) & (y > threshold_y)
    manders_x = x[idx].sum() / x.sum()
    manders_y = y[idx].sum() / y.sum()

    return pearson_r, r_prob, manders_x, manders_y


# def rolling_mean_filter(
#     x: np.ndarray, window: Tuple[int, int], threshold: int = 3
# ) -> np.ndarray:
#     """Rolling mean filter an array.

#     The window size should be an integer divisor of the array size.

#     Args:
#         window: Shape of the rolling window.
#         threshold: σ's value must be from mean to be an outlier.

#     """
#     x = x.copy()
#     # Create view
#     roll = rolling_window(x, window)
#     # Distance from mean (in stdevs)
#     means = np.mean(roll, axis=(2, 3), keepdims=True)
#     stds = np.std(roll, axis=(2, 3), keepdims=True)
#     diffs = np.abs(roll - means) / stds
#     # Recalculate mean, without outliers
#     roll[diffs > threshold] = np.nan
#     means = np.nanmean(roll, axis=(2, 3), keepdims=True)
#     # Replace all outliers and copy back into view
#     np.copyto(roll, means, where=diffs > threshold)
#     return x


# def rolling_median_filter(
#     x: np.ndarray, window: Tuple[int, int], threshold: int = 3
# ) -> np.ndarray:
#     """Rolling median filter an array.

#     The window size should be an integer divisor of the array size.

#     Args:
#         window: Shape of the rolling window.
#         threshold: N-distance's from median to be considered outlier.

#     """
#     x = x.copy()
#     # Create view
#     roll = rolling_window(x, window)
#     # Distance from the median
#     medians = np.median(roll, axis=(2, 3), keepdims=True)
#     diffs = np.abs(roll - medians)
#     # Median difference
#     median_diffs = np.median(diffs, axis=(2, 3), keepdims=True)
#     # Normalise differences
#     diffs = np.divide(diffs, median_diffs, where=median_diffs != 0)
#     # Replace all over threshold and copy back into view
#     np.copyto(roll, medians, where=diffs > threshold)

#     return x


# def blocks_view(x: np.ndarray, blocksize: Tuple[int, int]) -> np.ndarray:
#     """Create view into a array of blocksize size.

#     Args:
#         x: The array.
#         blocksize: The size of the each block.

#     Returns:
#         An array of views.
#     """
#     x = np.ascontiguousarray(x)
#     shape = tuple(np.array(x.shape) // blocksize) + blocksize
#     strides = tuple(np.array(x.strides) * blocksize) + x.strides
#     return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def view_as_blocks(x: np.ndarray, block: Tuple[int, int]) -> np.ndarray:
    """Create non-overlapping views into a array.

    Args:
        x: The array.
        window: The size of the view.

    Returns:
        An array of views.
    """
    assert len(block) == x.ndim
    x = np.ascontiguousarray(x)
    shape = tuple(np.array(x.shape) // block) + tuple(block)
    strides = tuple(np.array(x.strides) * block) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def shuffle_tiles(x: np.ndarray, block: Tuple[int, int]) -> np.ndarray:
    """Shuffle a 2d array as tiles of a certain size."""
    # Pad the array to fit the blocksize
    px = block[0] - x.shape[0] % block[0]
    py = block[1] - x.shape[1] % block[1]
    blocks = view_as_blocks(np.pad(x, ((0, px), (0, py)), mode="edge"), block)
    shape = blocks.shape

    # Reshape to (x, blocksize) and then shuffle
    blocks = blocks.reshape(-1, *block)
    np.random.shuffle(blocks)

    # Reform the image and then trim off excess
    return np.hstack(np.hstack(blocks.reshape(shape)))[: x.shape[0], : x.shape[1]]


# def rolling_window_step(
#     x: np.ndarray, window: Tuple[int, int], step: int
# ) -> np.ndarray:
#     """Create overlapping views into a array.

#     Args:
#         x: The array.
#         window: The size of the view.
#         step: Offset of the next window.

#     Returns:
#         An array of views.
#     """
#     x = np.ascontiguousarray(x)
#     slices = tuple(slice(None, None, st) for st in (step,) * x.ndim)
#     shape = tuple(
#         list(((np.array(x.shape) - np.array(window)) // np.array(step)) + 1)
#         + list(window)
#     )
#     strides = tuple(list(x[slices].strides) + list(x.strides))
#     return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
