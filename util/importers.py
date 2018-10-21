import numpy as np
import os


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
    return layer[list(layer.dtype.names[1:])]  # Remove times
