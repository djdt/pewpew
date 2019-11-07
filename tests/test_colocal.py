import numpy as np
import pytest

from pewpew.lib import colocal


a = np.tile([[0.0, 1.0], [0.0, 1.0]], 10)
b = np.tile([[0.0, 1.0], [1.0, 0.0]], 10)
c = np.tile([[1.0, 0.0], [1.0, 0.0]], 10)
d = np.tile([[1., 2.], [3., 4.]], 10)
e = np.tile([[1., 2.], [4., 3.]], 10)


def test_pearson_r():
    assert colocal.pearsonr(a, a) == 1.0
    assert colocal.pearsonr(a, b) == 0.0
    assert colocal.pearsonr(a, c) == -1.0


def test_pearson_r_probability():
    assert colocal.pearsonr_probablity(a, b, block=2, n=100) == (0.0, 1.0)


def test_li_icq():
    assert colocal.li_icq(a, a) == 0.5
    assert colocal.li_icq(a, b) == 0.0
    assert colocal.li_icq(a, c) == -0.5


def test_manders():
    assert colocal.manders(a, a, 0, 0) == (1.0, 1.0)
    assert colocal.manders(a, b, 0, 0) == (0.5, 0.5)
    assert colocal.manders(a, c, 0, 0) == (0.0, 0.0)


def test_costes_threshold():
    assert pytest.approx(colocal.costes_threshold(a, b), (1.0, 0.0, 0.5))
