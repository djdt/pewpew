import numpy as np
import os
import re

from util.laser import LaserData


def importNpz(path, config_override=None):
    lds = []
    npz = np.load(path)

    # Get the data
    # Create an empty array with correct dtype
    # Add data

    for datatype, isotope, config in zip(
            npz['datatypes'], npz['isotopes'], npz['configs']):
        lds.append(datatype(
            config=config_override if config_override is not None else config,
            isotope=isotope, data=npz[isotope], source=path))
    return lds


def importAgilentBatch(path, config):
    data_files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.lower().endswith('.d') and entry.is_dir():
                file_name = entry.name[:entry.name.rfind('.')] + '.csv'
                data_files.append(os.path.join(entry.path, file_name))
    # Sort by name
    data_files.sort()

    with open(data_files[0], 'r') as fp:
        line = fp.readline()
        skip_header = 0
        while line and not line.startswith('Time [Sec]'):
            line = fp.readline()
            skip_header += 1

        skip_footer = 0
        if "Print" in fp.read().splitlines()[-1]:
            skip_footer = 1

    cols = np.arange(1, line.count(',') + 1)

    lines = [np.genfromtxt(f, delimiter=',', names=True, usecols=cols,
             skip_header=skip_header, skip_footer=skip_footer)
             for f in data_files]
    # layer = np.vstack(lines)

    # lds = []
    # for name in layer.dtype.names:
    #     lds.append(LaserData(isotope=name, config=config, source=path,
    #                          data=layer[name]))
    return LaserData(data=np.vstack(lines), config=config, source=path)


def importCSVFromThatGermanThing(path):
    datare = re.compile(r'MainRuns;\d+;(\d+\w+);Counter;(.*)')

    isotopes = []
    data = {}
    with open(path, 'r') as fp:
        for line in fp.readlines():
            m = datare.match(line)
            if m is not None:
                i = m.group(1)
                linedata = np.fromstring(m.group(2), sep=';', dtype=float)
                isotopes.append(i)
                if i not in data.keys():
                    data[i] = []
                data[i].append(linedata)

    # Read the keys to ensure order is same
    keys = list(data.keys())
    # Stack lines to form 2d
    for k in keys:
        data[k] = np.vstack(data[k]).transpose()
    # Build a named array out of data
    dtype = [(k, float) for k in keys]
    structured = np.empty(data[keys[0]].shape, dtype)
    for k in keys:
        structured[k] = data[k]
    return structured
