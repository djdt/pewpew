import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class KMeansResult(object):
    """Stores the result of k-means and k-medians clustering.

    Args:
        k: number of clusters
        x: data that clustering was performed on
        labels: cluster mapping indicies
        centers: cluster centers

    Parameters:
        withinss: sum of square within cluster error
        totalss: sum of `withinss`
    """

    def __init__(self, k: int, x: np.ndarray, labels: np.ndarray, centers: np.ndarray):
        self.k = k
        self.labels = labels
        self.centers = centers
        self.withinss = np.array(
            [
                np.sum((x[self.labels == i] - self.centers[i]) ** 2)
                for i in range(self.k)
            ]
        )

    @property
    def totalss(self) -> float:
        return np.sum(self.withinss)


def kmeans_plus_plus(x: np.ndarray, k: int) -> np.ndarray:
    """Selects inital cluster positions using K-means++ algorithm.

    Args:
        x: data of shape (samples, features)
        k: number of clusters

    Returns:
        optimised initial cluster centers
    """
    ix = np.arange(x.shape[0])
    centers = np.empty((k, *x.shape[1:]))
    centers[0] = x[np.random.choice(ix, 1)]

    for i in range(1, k):
        distances = np.sum((centers[:i, None] - x) ** 2, axis=2)
        distances = np.amin(distances, axis=0)
        centers[i] = x[np.random.choice(ix, 1, p=distances / distances.sum())]

    return centers


def kcluster(
    x: np.ndarray,
    func: Callable[[np.ndarray, int], np.ndarray],
    k: int,
    init: str = "kmeans++",
    max_iterations: int = 1000,
) -> KMeansResult:
    """N-dim k- clustering

    Performs k- clustering of `x`, minimising intra-cluster variation.
    Better cluster starting positions can found by passing 'kmeans++' to `init`.

    Args:
        x: data of shape (samples, features)
        k: number of clusters
        init: initial cluster method, can be 'kmeans++' or 'random'
        max_iterations: maximum iterations for clustering

    Raises:
        ValueError if loop exceeds `max_iterations`

    See Also:
        :func:`pewpew.lib.kmeans.kmeans_plus_plus`
        :func:`pewpew.lib.kmeans.kmeans`
        :func:`pewpew.lib.kmeans.kmedians`
    """
    # Ensure at least 1 dim for variables
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if init == "kmeans++":
        centers = kmeans_plus_plus(x, k)
    elif init == "random":
        ix = np.random.choice(np.arange(x.shape[0]), k)
        centers = x[ix].copy()
    else:  # pragma: no cover
        raise ValueError("'init' must be 'kmeans++' or 'random'.")

    while max_iterations > 0:
        # Sort centers by the first attribute
        centers = centers[np.argsort(centers[:, 0])]

        distances = np.sqrt(np.sum((centers[:, None] - x) ** 2, axis=2))
        idx = np.argmin(distances, axis=0)

        new_centers = centers.copy()
        for i in range(k):
            new_centers[i] = func(x[idx == i], axis=0)

        if np.allclose(centers, new_centers):
            return KMeansResult(k, x, idx, centers)
        else:
            centers = new_centers
        max_iterations -= 1

    raise ValueError("No convergance in allowed iterations.")  # pragma: no cover


def kmeans(
    x: np.ndarray,
    k: int,
    init: str = "kmeans++",
    max_iterations: int = 1000,
) -> KMeansResult:
    """N-dim k-means clustering

    Performs k-means clustering of `x`, minimising intra-cluster variation.
    Better cluster starting positions can found by passing 'kmeans++' to `init`.

    Args:
        x: data of shape (samples, features)
        k: number of clusters
        init: initial cluster method, can be 'kmeans++' or 'random'
        max_iterations: maximum iterations for clustering

    Raises:
        ValueError if loop exceeds `max_iterations`

    See Also:
        :func:`pewpew.lib.kmeans.kmeans_plus_plus`
        :func:`pewpew.lib.kmeans.kmedians`
    """
    return kcluster(x, np.mean, k, init, max_iterations)


def kmedians(
    x: np.ndarray,
    k: int,
    init: str = "kmeans++",
    max_iterations: int = 1000,
) -> KMeansResult:
    """N-dim k-medians clustering

    Performs k-medians clustering of `x`, minimising intra-cluster variation.
    Better cluster starting positions can found by passing 'kmeans++' to `init`.

    Args:
        x: data of shape (samples, features)
        k: number of clusters
        init: initial cluster method, can be 'kmeans++' or 'random'
        max_iterations: maximum iterations for clustering

    Raises:
        ValueError if loop exceeds `max_iterations`

    See Also:
        :func:`pewpew.lib.kmeans.kmeans_plus_plus`
        :func:`pewpew.lib.kmeans.kmeans`
    """
    return kcluster(x, np.median, k, init, max_iterations)
