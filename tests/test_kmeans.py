import numpy as np

from pewpew.lib import kmeans


def test_kmeans_1d():
    np.random.seed(1983754)
    x = np.random.random(100)
    c = np.random.permutation(100)
    x[c[:25]] += 2.0

    idx = kmeans.kmeans1d(x, 2, method="kmeans", method_kws=dict(init="random"))
    _, counts = np.unique(idx, return_counts=True)
    assert np.all(np.sort(counts) == [25, 75])

    x[c[25:75]] -= 2.0

    idx = kmeans.kmeans1d(x, 3, method="kmeans", method_kws=dict(init="kmeans++"))
    _, counts = np.unique(idx, return_counts=True)
    assert np.all(np.sort(counts) == [25, 25, 50])


def test_kmeans_2d():
    np.random.seed(4923840)

    x = np.random.normal(loc=-2.5, size=100).reshape(50, 2)
    x[:, 0] = np.random.normal(loc=2.5, size=50)

    idx = kmeans.kmeans(x.ravel(), 2, init="kmeans++")
    _, counts = np.unique(idx, return_counts=True)
    assert np.all(np.sort(counts) == [50, 50])


def test_kmeans_threshold():
    x = np.zeros(100)
    x[:25] += 1.0
    x[:50] += 1.0
    x[:75] += 1.0

    t = kmeans.thresholds(x, 4)
    assert np.allclose(t, [1.0, 2.0, 3.0])
