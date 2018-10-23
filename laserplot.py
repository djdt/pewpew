import argparse
import sys
import matplotlib.pyplot as plt
from util.laser import Laser
from matplotlib_scalebar.scalebar import ScaleBar
from util.formatter import isotopeFormat
from util.laserfig import LaserFig


def parse_args(args):
    parser = argparse.ArgumentParser("LaserPlot")

    parser.add_argument('batchdir', help='Agilent batch directory (.b).')
    parser.add_argument('-i', '--isotope', default=None,
                        help='isotope to plot.')
    parser.add_argument('--cmap', default='magma',
                        choices=['viridis', 'plasma', 'inferno', 'magma'],
                        help='Colormap to use.')
    return vars(parser.parse_args(args))


def main(args):
    laser = Laser()
    laser.importData(args['batchdir'], importer='Agilent')

    if args['isotope'] is None:
        args['isotope'] = laser.getIsotopes()[0]

    # data = np.repeat(layer[args['isotope']], magfactor, axis=0)
    data = laser.getData(args['isotope'])
    fig = LaserFig(plt.gcf())

    fig.update(data, label=args['isotope'], extent=laser.getExtent(),
               aspect=laser.getAspect())

    # plt.imshow(data, cmap=args['cmap'], interpolation='none',
    #            extent=laser.getExtent(), aspect=laser.getAspect())
    # plt.axis('off')
    # plt.colorbar()

    # scalebar = ScaleBar(1.0, 'um', color='white',
    #                     frameon=False, location='lower right')
    # plt.gca().add_artist(scalebar)
    # plt.gca().annotate(isotopeFormat(args['isotope']),
    #                    xy=(0.05, 0.95), xytext=(5, -5),
    #                    xycoords='axes fraction', textcoords='offset pixels',
    #                    va='top', color='white')

    # plt.tight_layout()
    # plt.show()
    fig.show()
    plt.show()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
