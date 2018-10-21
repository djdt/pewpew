import numpy as np
import os


class AgilentImporter(object):
    def __init__(self):
        pass

    def getLayer(batch_dir):
        """Searches batch_dir (.b) or data files (.d/.csv), sorts by name
           and builds layer."""

        data_files = []
        with os.scandir(batch_dir) as it:
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

    # def checkParams(self, params):
    #     """Returns True if params are valid."""
    #     if params.krisskross:
    #         for i in range(0, len(self.layers) - 1):
    #             if self.layers[i+1].shape[1] < params.warmup + \
    #                self.layers[i].shape[0] * params.magfactor:
    #                 return False
    #     else:
    #         for layer in self.layers:
    #             if layer.shape[1] < params.warmup + params.linescans:
    #                 return False

    #     return True
