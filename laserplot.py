import argparse
import sys
import matplotlib.pyplot as plt
from util.laser import Laser
from util.laserimage import LaserImage
import numpy as np


def parse_args(args):
    parser = argparse.ArgumentParser("LaserPlot")

    parser.add_argument('batchdir', help='Agilent batch directory (.b).')
    parser.add_argument('-i', '--isotopes', nargs='+', default=None,
                        help='isotope to plot.')
    parser.add_argument('--export', action='store_true',
                        help='Export selected isotopes to .npy.')
    parser.add_argument('--list', action='store_true',
                        help='List isotopes and exit.')
    parser.add_argument('--scantime', default=0.25, type=float)
    parser.add_argument('--speed', default=120.0, type=float)
    parser.add_argument('--spotsize', default=30.0, type=float)
    parser.add_argument('--gradient', default=1.0, type=float)
    parser.add_argument('--intercept', default=0.0, type=float)
    parser.add_argument('--cmap', default='magma',
                        help='Colormap to use.')
    return vars(parser.parse_args(args))


def main(args):
    laser = Laser(spotsize=args['spotsize'], speed=args['speed'],
                  scantime=args['scantime'], gradient=args['gradient'],
                  intercept=args['intercept'])
    laser.importData(args['batchdir'], importer=args['importer'])

    if args['list']:
        print('Isotopes:')
        for i in laser.getIsotopes():
            print('\t' + i)
        sys.exit(0)

    if args['isotopes'] is None:
        args['isotopes'] = laser.getIsotopes()

    if args['export']:
        print('Exporting:')
        for i in args['isotopes']:
            np.save(args['batchdir'].rstrip('/').replace('.b', f'.{i}.npy'),
                    laser.getData(i))
        sys.exit(0)

    fig, axes = plt.subplots(1, len(args['isotopes']))

    if len(args['isotopes']) == 1:
        axes = [axes]

    for ax, label in zip(axes, args['isotopes']):
        print(label)
        data = laser.getData(label)
        LaserImage(fig, ax, data, label=label,
                   extent=laser.getExtent(), aspect=laser.getAspect(),
                   cmap=args['cmap'])

    plt.show()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
