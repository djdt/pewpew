import argparse
import sys
import matplotlib.pyplot as plt
from util.laser import Laser
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np


def parse_args(args):
    parser = argparse.ArgumentParser("LaserPlot")

    parser.add_argument('batchdir', help='Agilent batch directory (.b).')
    parser.add_argument('-e', '--element', default=None,
                        help='Element to plot.')
    parser.add_argument('--cmap', default='magma',
                        choices=['viridis', 'plasma', 'inferno', 'magma'],
                        help='Colormap to use.')
    return vars(parser.parse_args(args))


def main(args):
    laser = Laser()
    laser.importData(args['batchdir'], importer='Agilent')

    if args['element'] is None:
        args['element'] = laser.getElements()[0]

    # data = np.repeat(layer[args['element']], magfactor, axis=0)
    data = laser.getData(args['element'])

    plt.imshow(data, cmap=args['cmap'], interpolation='none',
               extent=laser.getExtent(), aspect=laser.getAspect())
    plt.axis('off')
    plt.colorbar()

    scalebar = ScaleBar(1.0, 'um', color='white',
                        frameon=False)
    plt.gca().add_artist(scalebar)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
