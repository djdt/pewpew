import numpy as np

from pewpew.lib import kmeans


def test_kmeans_1d():
    np.random.seed(1983754)
    x = np.array(
        [9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    )

    result = kmeans.kmeans(x.ravel(), 2, init="random")
    assert result.k == 2
    assert np.all(
        result.labels
        == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    assert np.all(result.centers == [[2.0], [8.0]])
    assert np.all(result.withinss == [8.0, 8.0])
    assert result.totalss == 16.0
    result = kmeans.kmeans(x, 2, init="kmeans++")
    assert result.k == 2
    assert np.all(
        result.labels
        == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    assert np.all(result.centers == [[2.0], [8.0]])
    assert np.all(result.withinss == [8.0, 8.0])
    assert result.totalss == 16.0


def test_kmeans_1d_sin():
    x = np.sin(np.linspace(np.pi / 200.0, np.pi * 2, 200))
    center = np.mean(x[:100])
    result = kmeans.kmeans(x.ravel(), 2, init="kmeans++")
    assert np.all(result.labels[:100] == 1)
    assert np.all(result.labels[100:] == 0)
    assert np.all(np.isclose(result.centers, [[-center], [center]], rtol=1e-3))


def test_kmeans_2d():
    np.random.seed(1983754)
    x = np.stack(
        ([3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0], [5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7]),
        axis=1,
    )

    result = kmeans.kmeans(x, 3, init="kmeans++")
    assert result.k == 3
    assert np.all(result.labels == [2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    assert np.all(result.centers == [[0.0, 7.0], [1.5, 6.0], [3.0, 5.0]])
    assert np.all(result.withinss == [0.0, 1.5, 0.0])
    assert result.totalss == 1.5


def test_kmedians():
    np.random.seed(1983754)
    x = np.array([1.0, 1.0, 6.0, 10.0, 50.0, 60.0, 61.0, 90.0])
    result = kmeans.kmeans(x, 3, init="kmeans++")
    assert np.all(result.labels == [0, 0, 1, 1, 2, 2, 2, 2])
    result = kmeans.kmedians(x, 3, init="kmeans++")
    assert np.all(result.labels == [0, 0, 0, 0, 1, 1, 1, 2])


def test_kmeans_threshold():
    x = np.zeros(100)
    x[:25] += 1.0
    x[:50] += 1.0
    x[:75] += 1.0

    t = kmeans.thresholds(x, 4)
    assert np.allclose(t, [1.0, 2.0, 3.0])
