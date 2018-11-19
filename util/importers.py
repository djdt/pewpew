import numpy as np
import os
import re


def importAgilentBatch(path):
    data_files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith('.d') and entry.is_dir():
                file_name = entry.name.replace('.d', '.csv')
                data_files.append(os.path.join(entry.path, file_name))
    # Sort by name
    data_files.sort()

    lines = [np.genfromtxt(
             f, delimiter=',', names=True,
             skip_header=3, skip_footer=1) for f in data_files]
    layer = np.vstack(lines)
    # return layer[list(layer.dtype.names[1:])]  # Remove times
    return layer[list(layer.dtype.names[1:])]  # Remove times

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

if __name__ == "__main__":

    f = "/home/tom/Downloads/20180509_her2_1.csv"
    data = importCSVFromThatGermanThing(f)
    for i in data.dtype.names:
        np.save(f"{os.path.basename(f)}.{i}.npy", data[i])

