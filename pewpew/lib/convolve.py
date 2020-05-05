import numpy as np

_s2 = np.sqrt(2.0)
_s2pi = np.sqrt(2.0 * np.pi)


def deconvolve(x: np.ndarray, psf: np.ndarray, mode: str = "valid"):
    """From https://rosettacode.org/wiki/Deconvolution/1D"""
    def shift_bit_length(x: int) -> int:
        return 1 << (x - 1).bit_length()
    r = shift_bit_length(max(x.size, psf.size))
    y = np.fft.irfft(np.fft.rfft(x, r) / np.fft.rfft(psf, r), r)
    rec = np.trim_zeros(np.real(y))[:x.size - psf.size - 1]
    if mode == "valid":
        return rec
    elif mode == "same":
        return np.hstack((rec, x[rec.size:]))
    else:
        raise ValueError("Valid modes are 'valid', 'same'.")


def erf(x: float) -> float:
    """Error function. Maximum error is 5e-4.
    From 'Abramowitz and Stegun'.
    """
    assert x >= 0.0
    # Maximum error: 2.5e-5
    a = np.array([0.278393, 0.230389, 0.000972, 0.078108])
    x = np.power(x, [1, 2, 3, 4])

    return 1.0 - 1.0 / np.power(1.0 + np.sum(a * x), 4)


def erfinv(x: float) -> float:
    """Inverse error function. Maximum error is 6e-3.
    From 'A handy approximation for the error function and its inverse' by Sergei Winitzki
    """
    sign = np.sign(x)
    x = np.log((1.0 - x) * (1.0 + x))

    tt1 = 2.0 / (np.pi * 0.14) + 0.5 * x
    tt2 = 1.0 / 0.14 * x

    return sign * np.sqrt(-tt1 + np.sqrt(tt1 * tt1 - tt2))


def gaussian_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    return 1.0 / (sigma * _s2pi) * np.exp(-0.5 * np.power(x / sigma, 2))


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    x = np.linspace(gaussian_quantile(0.01, 1.0), gaussian_quantile(0.99, 1.0), size)
    kernel = gaussian_pdf(x, sigma)
    return kernel / kernel.sum()


def gaussian_quantile(quantile: float, sigma: float) -> float:
    return sigma * _s2 * erfinv(2.0 * quantile - 1.0)


def log_normal_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    return 1.0 / (x * sigma * _s2pi) * np.exp(-0.5 * np.power(np.log(x) / sigma, 2))


def log_normal_psf(size: int, sigma: float) -> np.ndarray:
    x = np.linspace(
        log_normal_quantile(0.01, sigma), log_normal_quantile(0.99, sigma), size
    )
    kernel = log_normal_pdf(x, sigma)
    return kernel / kernel.sum()


def log_normal_quantile(quantile: float, sigma: float) -> float:
    return np.exp(np.sqrt(2.0 * sigma * sigma) * erfinv(2.0 * quantile - 1.0))
