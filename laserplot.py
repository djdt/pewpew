import argparse
import sys
import matplotlib.pyplot as plt
from util.importers import AgilentImporter


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
    layer = AgilentImporter.getLayer(args['batchdir'])

    if args['element'] is None:
        args['element'] = layer.dtype.names[0]

    plt.imshow(layer[args['element']], cmap=args['cmap'],
               interpolation='none')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
