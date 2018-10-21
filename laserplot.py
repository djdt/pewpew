import argparse
import sys
import matplotlib.pyplot as plt
from util.importers import importAgilentBatch
import matplotlib_scalebar.scalebar as ScaleBar


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

    x = 138 * 30

    if args['element'] is None:
        args['element'] = layer.dtype.names[0]

    plt.imshow(layer[args['element']], cmap=args['cmap'],
               interpolation='none',
               extent=(0, x, 0, x / 5))
    plt.axis('off')
    plt.colorbar()

    scalebar = ScaleBar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
