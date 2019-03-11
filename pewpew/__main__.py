import sys

from PyQt5.QtWidgets import QApplication
from pewpew.ui import MainWindow



if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    sys.excepthook = window.exceptHook  # type: ignore
    window.show()

    app.exec()
    # import numpy as np

    # def fft_filter(x: np.ndarray, threshold: float = 0.1, amplitude: float = 0.01) -> None:
    #     # x[:] = np.fft.fftshift(np.fft.fft2(x)[:])
    #     fft = np.fft.fftshift(np.fft.fft2(x))
    #     outlier = x.where(fft > amplitude)
    #     # outlier = np.max(x) if np.abs(np.max(x) > np.abs(np.min(x))) else np.min(x)
    #     # if np.any(np.abs(fft[fft > threshold]) > amplitude):
    #     #     index = np.where(x == outlier)
    #     #     x[index] = 1000000000

    # from pewpew.lib.io import npz

    # from pewpew.lib.calc import rolling_median_filter
    # ld = npz.load("/home/tom/Dropbox/Uni/Experimental/LAICPMS/agilent/20190124_mn_stds.npz")[0]

    # import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(2, 2)

    # x = ld.data["57Fe"]
    # fft = np.fft.fft2(x)
    # # f = np.fft.fftfreq(fft.size, 0.001)

    # axes[0, 0].imshow(x)
    # axes[0, 1].imshow(np.abs(fft))

    # # outlier = fft[np.all(f > 0.1, fft > 0.001)]
    # mean = x.copy()
    # rolling_median_filter(mean, (3, 3))
    # axes[1, 0].imshow(mean)

    # r, c = fft.shape
    # # fixed = np.zeros_like(fft)
    # # fixed[int(r * 0.35):int(r * 0.65), int(c * 0.35):int( c * 0.65)] = fft[int(r * 0.35):int(r * 0.65), int(c * 0.35):int( c * 0.65)]
    # # fixed = fft
    # # fixed[int(r * 0.35):int(r * 0.65), int(c * 0.35):int( c * 0.65)] = 0
    # # axes[1, 1].imshow(np.abs(np.fft.ifft2(np.fft.fftshift(fixed))))

    # plt.show()
