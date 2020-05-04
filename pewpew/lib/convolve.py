import numpy as np


def gaussian_kernel(size: int, sigma: float, power: int = 1) -> np.ndarray:
    r = 2.3263478740  # 0.99 quantile
    x = np.linspace(-r, r, size)
    kernel = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power(np.power(x / sigma, 2), power))
    return kernel


def gaussian_psf(size: int, sigma: float, power: int = 1) -> np.ndarray:
    r = 2.3263478740  # 0.99 quantile
    x = np.linspace(-r, r, size)
    kernel = np.exp(-0.5 * np.power(np.power(x / sigma, 2), power))
    return kernel / kernel.sum()
