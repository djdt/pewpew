import matplotlib.pyplot as plt

from pewpew.lib.mpltools import MetricSizeBar


def test_metric_size_bar():
    fig, axes = plt.subplots(5)

    facecolors = ["white", "black", "white", "black", "white"]
    locations = [1, 2, 3, 4, 5]
    limits = [(0, 1e-13), (0, 2e2), (0, 3e3), (0, 4e6), (0, 1e13)]
    sizebars = []

    for ax, fc, loc, lim in zip(axes, facecolors, locations, limits):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_facecolor(fc)
        ax.set_xlim(*lim)
        sizebar = MetricSizeBar(ax, "μm", loc=loc)
        ax.add_artist(sizebar)
        sizebars.append(sizebar)

    plt.ion()
    plt.show()
    plt.pause(0.001)

    assert sizebars[0].txt_label.get_text() == "No Scale"
    assert sizebars[1].txt_label.get_text() == "20 μm"
    assert sizebars[2].txt_label.get_text() == "500 μm"
    assert sizebars[3].txt_label.get_text() == "50 cm"
    assert sizebars[4].txt_label.get_text() == "No Scale"

    plt.close()
