import numpy as np

from .calc import normalise, shuffle_tiles

from typing import Tuple


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Returns Pearson's colocalisation coefficient for the two arrays.
    This value for colocalisation between -1 and 1 (colocalised).
"""
    return (np.nanmean(x * y) - (np.nanmean(x) * np.nanmean(y))) / (
        np.nanstd(x) * np.nanstd(y)
    )


def pearsonr_probablity(
    x: np.ndarray, y: np.ndarray, blocksize: int = 5, n: int = 1000
) -> Tuple[float, float]:
    """Returns Pearson's colocalisation coefficient and the relevant probabilty.
"""
    r = pearsonr(x, y)
    rs = np.array(
        [pearsonr(x, shuffle_tiles(y, (blocksize, blocksize))) for i in range(n)]
    )
    return r, (rs < r).sum() / n


def manders(
    x: np.ndarray, y: np.ndarray, tx: float, ty: float = None
) -> Tuple[float, float]:
    """Returns Manders' correlation coefficients m1, m2.
    tx and ty are the thresholds for x and y respectively.
    If ty is None then tx is used.
"""
    if ty is None:
        ty = tx

    return np.sum(x, where=x > tx) / x.sum(), np.sum(y, where=y > ty) / y.sum()


def costes_threshold(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Calculates the thresholds Tx and Ty for the given arrays.
    Arrays should be normalised before calling this function.

    Returns:
        T -> threshold for x
        a -> slope
        b -> interept
"""
    b, a = np.polynomial.Polynomial.fit(x.ravel(), y.ravel(), 1).convert().coef
    threshold = x.max()
    threshold_min = x.min()
    increment = (threshold - threshold_min) / 256.0

    idx = np.logical_or(x <= threshold, y <= (a * threshold + b))
    r = pearsonr(x[idx], y[idx])

    while r > 0.0 and threshold > threshold_min:
        threshold -= increment
        idx = np.logical_or(x <= threshold, y <= (a * threshold + b))
        if np.all(x[idx] == 0) or np.all(y[idx] == 0):
            threshold = threshold_min
            break
        r = pearsonr(x[idx], y[idx])

    return threshold, a, b


def costes(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    x, y = normalise(x), normalise(y)
    pearson_r, r_prob = pearsonr_probablity(x, y, n=200)
    T, a, b = costes_threshold(x, y)
    m1, m2 = manders(x, y, T, a * T + b)

    return pearson_r, r_prob, m1, m2
