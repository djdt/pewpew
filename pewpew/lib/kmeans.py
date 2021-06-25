import numpy as np
import logging

logger = logging.getLogger(__name__)


class KMeansResult(object):
    def __init__(self, k: int, x: np.ndarray, labels: np.ndarray, centers: np.ndarray):
        self.k = k
        self.data = x
        self.labels = labels
        self.centers = centers

    def totalss(self) -> float:
        return np.sum(self.withinss)

    def withinss(self) -> np.ndarray:
        return np.array(
            [
                np.sum((self.data[self.labels == i] - self.centers[i]) ** 2)
                for i in range(self.k)
            ]
        )


def kmeans_plus_plus(x: np.ndarray, k: int) -> np.ndarray:
    """Selects inital cluster positions using K-means++ algorithm.

    Args:
        x: nd array
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
       x: shape of (n, m) for n objects with m attributes
       k: number of clusters
       init: initial cluster method Can be 'kmeans++' or 'random'
       max_iterations: maximum iterations for clustering

    Returns:
        KMeansResult, object containing k, labels, centers

    Raises:
        ValueError if loop exceeds `max_iterations`

    See Also:
        :func:`pewpew.lib.kmeans.kmeans_plus_plus`
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
            new_centers[i] = np.mean(x[idx == i], axis=0)

        if np.allclose(centers, new_centers):
            return KMeansResult(k, x, idx, centers)
        else:
            centers = new_centers
        max_iterations -= 1

    raise ValueError("No convergance in allowed iterations.")  # pragma: no cover


def kmeans1d(
    x: np.ndarray, k: int, method: str = "ckmeans1d", method_kws: dict = None
) -> np.ndarray:
    """1-dim k-means clustering.
    Uses Ckmeans.1d.dp through ``ckwrap`` it is is installed and `method` is
    'ckmeans1d'.

    Args:
        x: flattened to 1d
        k: number of clusters
        method: if 'ckmeans1d' ckwrap is used, otherwise 'kmeans' in 1d
        method_kws: passed through to the implementaion used

    Returns:
        array of labels mapping clusters to objects

    See Also:
        :func:`pewpew.lib.kmeans.kmeans`
    """
    kwargs = {
        "init": "kmeans++",
        "max_iterations": 1000,
        "weights": None,
        "method": "linear",
    }
    if method_kws is not None:
        kwargs.update(method_kws)

    if method == "ckmeans1d":  # pragma: no cover
        try:
            from ckwrap import ckmeans

            idx = ckmeans(
                x.ravel(),
                (k, k),
                weights=kwargs["weights"],
                method=kwargs["method"],
            ).labels
        except ImportError:
            logger.warning("Unable to use ckmeans1d as ckwrap package not found.")
            return kmeans1d(x, k, method="kmeans", method_kws=method_kws)
    elif method == "kmeans":
        idx = kmeans(
            x.ravel(),
            k,
            init=kwargs["init"],  # type: ignore
            max_iterations=kwargs["max_iterations"],  # type: ignore
        ).labels
    else:  # pragma: no cover
        raise ValueError(f"Unknown method {method}.")
    return np.reshape(idx, x.shape)


def thresholds(x: np.ndarray, k: int) -> np.ndarray:
    """Produces thresholds from minimum cluster values.

    Uses k-means clustering to group array into k clusters and produces k - 1
    thresholds using the minimum value of each cluster.
    """
    idx = kmeans1d(x, k)
    return np.array([np.amin(x[idx == i]) for i in range(1, k)])
