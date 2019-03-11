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

    # ld = npz.load("/home/tom/Dropbox/Uni/Experimental/LAICPMS/agilent/20190124_mn_stds.npz")[0]

    # import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(2, 2)

    # x = ld.data["57Fe"]
    # fft = np.fft.fftshift(np.fft.fft2(x))

    # axes[0, 0].imshow(x)
    # axes[0, 1].imshow(np.abs(fft))

    # outlier = np.log(fft) > 13
    # axes[1, 0].imshow(outlier)

    # fft[fft > 13] = 0
    # axes[1, 1].imshow(np.abs(np.fft.ifft2(np.fft.fftshift(fft))))

    # plt.show()
