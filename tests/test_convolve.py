import numpy as np
import pytest

from pewpew.lib import convolve


# The majority of these tests are comparing against results from scipy.stats


def test_deconvolve():

    x = np.random.randint(0, 100, 50)
    k = np.array([0.6, 0.3, 0.1])

    c = np.convolve(x, k, mode="full")
    d = convolve.deconvolve(c, k, mode="valid")

    assert np.allclose(x[:-2], d)


def test_functions():
    pytest.approx(convolve.gamma(1.0), 1.0)
    pytest.approx(convolve.gamma(0.1), 9.5135)

    pytest.approx(convolve.erf(1.0), 0.84270)

    pytest.approx(convolve.erfinv(convolve.erf(1.0)), 1.0)
    pytest.approx(convolve.erfinv(convolve.erf(2.0)), 2.0)
    pytest.approx(convolve.erfinv(convolve.erf(3.0)), 3.0)


def test_kernels():
    assert np.allclose(
        convolve.beta(10, 1.0, 2.0)[:, 1],
        [
            0.2,
            0.17777778,
            0.15555556,
            0.13333333,
            0.11111111,
            0.08888889,
            0.06666667,
            0.04444444,
            0.02222222,
            0.0,
        ],
    )
    assert np.allclose(
        convolve.exponential(10, 1.0)[:, 1],
        [
            6.70817001e-01,
            2.20828277e-01,
            7.26951285e-02,
            2.39307292e-02,
            7.87782913e-03,
            2.59332640e-03,
            8.53704959e-04,
            2.81033718e-04,
            9.25143394e-05,
            3.04550752e-05,
        ],
    )
    assert np.allclose(
        convolve.inversegamma(10, 1.0, 1.0)[:, 1],
        [
            0.0,
            0.50910576,
            0.19960919,
            0.10307235,
            0.06249378,
            0.04183695,
            0.02993825,
            0.02247186,
            0.01748376,
            0.0139881,
        ],
    )
    assert np.allclose(
        convolve.laplace(10, 1.0, 1.0)[:, 1],
        [
            0.00144871,
            0.0044008,
            0.01336846,
            0.0406098,
            0.12336168,
            0.37473971,
            0.3000681,
            0.09878031,
            0.03251779,
            0.01070463,
        ],
    )
    assert np.allclose(
        convolve.loglaplace(10, 0.5, 0.0)[:, 1],
        [
            1.14643277e-06,
            8.35747483e-01,
            1.04468576e-01,
            3.09536662e-02,
            1.30585809e-02,
            6.68599430e-03,
            3.86921001e-03,
            2.43658722e-03,
            1.63232316e-03,
            1.14643277e-03,
        ],
    )
    assert np.allclose(
        convolve.lognormal(10, 1.0, 0.0)[:, 1],
        [
            2.32134774e-36,
            5.80902822e-01,
            2.12338077e-01,
            9.43250504e-02,
            4.80058548e-02,
            2.68544041e-02,
            1.61003829e-02,
            1.01794286e-02,
            6.71278012e-03,
            4.58120083e-03,
        ],
    )
    assert np.allclose(
        convolve.normal(10, 1.0, 1.0)[:, 1],
        [
            6.75098856e-09,
            2.86141536e-06,
            3.52881109e-04,
            1.26622210e-02,
            1.32198138e-01,
            4.01582486e-01,
            3.54942629e-01,
            9.12799790e-02,
            6.83009705e-03,
            1.48700419e-04,
        ],
    )
    assert np.allclose(  # Arbitrary test
        convolve.super_gaussian(10, 1.0, 0.0, 2.0)[:, 1],
        [
            9.84282315e-137,
            1.10778978e-050,
            6.05005340e-014,
            1.08299498e-002,
            4.89170050e-001,
            4.89170050e-001,
            1.08299498e-002,
            6.05005340e-014,
            1.10778978e-050,
            9.84282315e-137,
        ],
    )
    assert np.allclose(
        convolve.triangular(10, -5.0, 5.0)[:, 1],
        [0.0, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.0],
    )


test_functions()
