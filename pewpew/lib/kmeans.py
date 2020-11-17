import numpy as np


def _kmeans_plus_plus(x: np.ndarray, k: int) -> np.ndarray:
    """Selects inital cluster positions using K-means++ algorithm."""
    ix = np.arange(x.shape[0])
    centers = np.empty((k, *x.shape[1:]))
    centers[0] = x[np.random.choice(ix, 1)]

    for i in range(1, k):
        distances = np.sqrt(np.sum((centers[:i, None] - x) ** 2, axis=2))
        distances = np.amin(distances, axis=0) ** 2
        centers[i] = x[np.random.choice(ix, 1, p=distances / distances.sum())]

    return centers.copy()


def kmeans(
    x: np.ndarray,
    k: int,
    init: str = "kmeans++",
    max_iterations: int = 1000,
) -> np.ndarray:
    """K-means clustering. Returns an array mapping objects to their clusters.
    Raises a ValueError if the loop exceeds max_iterations.

    Args:
       x: Data. Shape is (n, m) for n objects with m attributes.
       k: Number of clusters.
       init: Method to determine initial cluster centers. Can be 'kmeans++' or 'random'
    """
    # Ensure at least 1 dim for variables
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if init == "kmeans++":
        centers = _kmeans_plus_plus(x, k)
    elif init == "random":
        ix = np.random.choice(np.arange(x.shape[0]), k)
        centers = x[ix].copy()
    else:  # pragma: no cover
        raise ValueError("'init' must be 'kmeans++' or 'random'.")

    # Sort centers by the first attribute
    centers = centers[np.argsort((centers[:, 0]))]

    while max_iterations > 0:
        max_iterations -= 1

        distances = np.sqrt(np.sum((centers[:, None] - x) ** 2, axis=2))
        idx = np.argmin(distances, axis=0)

        new_centers = centers.copy()
        for i in np.unique(idx):
            new_centers[i] = np.mean(x[idx == i], axis=0)

        if np.allclose(centers, new_centers):
            return idx
        centers = new_centers

    raise ValueError("No convergance in allowed iterations.")  # pragma: no cover


def kmeans1d(x: np.ndarray, k: int, method: str = "ckmeans1d", **kwargs) -> np.ndarray:
    """K-means clustering. Returns an array mapping objects to their clusters.
    Uses ckwrap if it is available and 'non_optimal' is not False. Raises an error
    if 'use_ckwrap' is True and ckwrap is not found. **kwargs are passed to
    ckwrap.ckmeans or _kmeans.

    Args:
        x: Data. Flattened to 1d.
        k: number of clusters.
        method: If 'ckmeans1d' ckwrap is used, otherwise 'kmeans' in 1d.
        kwrags: kwargs passed through to the implementaion used.
    """
    if method == "ckmeans1d":
        try:
            from ckwrap import ckmeans

            idx = ckmeans(x.ravel(), (k, k), **kwargs).labels
        except ImportError:
            idx = kmeans(x.ravel(), k, **kwargs)
    return np.reshape(idx, x.shape)


def thresholds(x: np.ndarray, k: int) -> np.ndarray:
    """Uses k-means clustering to group array into k clusters and produces k - 1
    thresholds using the minimum value of each cluster."""
    idx = kmeans1d(x, k)
    return np.array([np.amin(x[idx == i]) for i in range(1, k)])
