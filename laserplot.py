import argparse
import sys
import matplotlib.pyplot as plt
from util.laser import LaserData, LaserParams
from util.laserimage import LaserImage
from util.importers import importAgilentBatch
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
    param = LaserParams(spotsize=args['spotsize'], speed=args['speed'],
                        scantime=args['scantime'], gradient=args['gradient'],
                        intercept=args['intercept'])

    laser = LaserData(importAgilentBatch(args['batchdir']), param)

    if args['list']:
        print('Isotopes:')
        for i in laser.isotopes():
            print('\t' + i)
        sys.exit(0)

    if args['isotopes'] is None:
        args['isotopes'] = laser.isotopes()

    if args['export']:
        print('Exporting:')
        for i in args['isotopes']:
            np.save(args['batchdir'].rstrip('/').replace('.b', f'.{i}.npy'),
                    laser.get(i))
        sys.exit(0)

    fig, axes = plt.subplots(1, len(args['isotopes']))

    if len(args['isotopes']) == 1:
        axes = [axes]

    for ax, label in zip(axes, args['isotopes']):
        print(label)
        data = laser.get(label)
        LaserImage(fig, ax, data, label=label,
                   extent=laser.extent(), aspect=laser.aspect(),
                   cmap=args['cmap'])

    plt.show()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
