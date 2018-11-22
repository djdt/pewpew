import numpy as np


def krissKrossLayers(self, layers, aspect, warmup, horizontal_first=True):

        j = 0 if horizontal_first else 1
        aspect = int(aspect)
        trim = int(aspect / 2)
        # Calculate the line lengths
        length = (layers[1].shape[0] * aspect,
                  layers[0].shape[0] * aspect)

        # Reshape the layers and stack into matrix
        transformed = []
        for i, layer in enumerate(layers):
            # Trim data of warmup time and excess
            layer = layer[:, warmup:warmup+length[(i + j) % 2]]
            # Stretch array
            layer = np.repeat(layer, aspect, axis=0)
            # Flip vertical layers and trim
            if (i + j) % 2 == 0:
                layer = layer.T
                layer = layer[trim:, trim:]
            elif trim > 0:
                layer = layer[:-trim, :-trim]

            transformed.append(layer)

        data = np.dstack(transformed)

        # TODO find a way to do this, less copy() required
        # if self.params['rastered']:
        #     self.data[1::2, :, 0::2] = self.data[1::2, ::-1, 0::2]
        #     self.data[:, 1::2, 1::2] = self.data[::-1, 1::2, 1::2]

        return data
