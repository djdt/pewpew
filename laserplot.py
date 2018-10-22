import argparse
import sys
import matplotlib.pyplot as plt
from util.importers import importAgilentBatch
# import matplotlib_scalebar.scalebar as ScaleBar
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
    print(args['batchdir'])
    layer = importAgilentBatch(args['batchdir'])

    print(layer.shape)

    if args['element'] is None:
        args['element'] = layer.dtype.names[0]

    pix_size = 0.25 * 120
    magfactor = 15.0 / (pix_size)
    print(pix_size)
    print(magfactor)


    # data = np.repeat(layer[args['element']], magfactor, axis=0)
    data = layer[args['element']]

    plt.imshow(data, cmap=args['cmap'],
               interpolation='none', extent=(0, data.shape[1] * pix_size, 0, data.shape[0] * 30.0))
    plt.axis('off')
    plt.colorbar()

    # scalebar = ScaleBar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
