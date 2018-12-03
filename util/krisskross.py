import numpy as np

from util.laser import LaserData


def krissKrossLayers(layers, aspect, warmup, horizontal_first=True):

    j = 0 if horizontal_first else 1
    aspect = int(aspect)
    trim = int(aspect / 2)
    # Calculate the line lengths
    length = (layers[1].shape[0] * aspect, layers[0].shape[0] * aspect)

    # Reshape the layers and stack into matrix
    transformed = []
    for i, layer in enumerate(layers):
        # Trim data of warmup time and excess
        layer = layer[:, warmup:warmup + length[(i + j) % 2]]
        # Stretch array
        layer = np.repeat(layer, aspect, axis=0)
        # Flip vertical layers and trim
        if (i + j) % 2 == 1:
            layer = layer.T
            layer = layer[trim:, trim:]
        elif trim > 0:
            layer = layer[:-trim, :-trim]

        transformed.append(layer)

    data = np.dstack(transformed)

    return data


class KrissKrossData(LaserData):
    def __init__(self, data=None, config=None, source=""):
        super().__init__(data=data, config=config, source=source)

    def fromLayers(self, layers, warmup_time=13.0, horizontal_first=True):
        warmup = int(warmup_time / self.config['scantime'])
        self.data = krissKrossLayers(layers, self.aspect(), warmup,
                                     horizontal_first)

    def calibrated(self, isotope=None, flatten=True):
        """Returns a flattened array (via mean) by default."""
        return np.mean(super().calibrated(isotope), axis=2)

    def split(self):
        lds = []
        for data in np.dsplit(self.data, self.data.shape[2]):
            # Strip the third dimension
            lds.append(
                KrissKrossData(
                    data=data, config=self.config, source=self.source))
        return lds

    def extent(self):
        # Image data is stored [rows][cols]
        x = self.data.shape[1] * self.pixelsize()[0]
        y = self.data.shape[0] * self.pixelsize()[1] / self.aspect()
        return (0, x, 0, y)

    def layers(self):
        return self.data.shape[2]
