# from util.importers import importAgilentBatch


class LaserParams(object):
    def __init__(self, scantime=0.25, speed=120.0, spotsize=30.0,
                 gradient=1.0, intercept=0.0):
        self.scantime = scantime  # s
        self.speed = speed        # um/s
        self.spotsize = spotsize  # um
        self.gradient = gradient
        self.intercept = intercept

    def pixelsize(self):
        return (self.speed * self.scantime, self.spotsize)

    def aspect(self):
        return 1.0 / ((self.speed * self.scantime) / self.spotsize)

    def extent(self, data_shape):
        x = data_shape[1] * self.pixelsize()[0]
        y = data_shape[0] * self.pixelsize()[1]
        return (0, x, 0, y)
