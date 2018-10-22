from util.importers import importAgilentBatch


class Laser(object):
    DEFAULT_PARAMS = {
        "scantime": 1.0,  # s
        "speed": 1.0,     # um/s
        "spotsize": 1.0,  # um
        "gradient": 1.0,
        "intercept": 1.0
    }

    def __init__(self):
        self.data = []

        self.scantime = 0.25  # s
        self.speed = 120.0    # um/s
        self.spotsize = 30.0  # um

        self.gradient = 1.0
        self.intercept = 0.0

    def getElements(self):
        return self.data.dtype.names

    def importData(self, path, importer='Agilent'):
        if importer is 'Agilent':
            self.data = importAgilentBatch(path)
        else:
            print(f'Laser.import: unknown importer \'{importer}\'!')

    def calibrate(self, gradient, intercept):
        self.gradient = gradient
        self.intercept = intercept

    def getData(self, element=None):
        if element is not None:
            data = self.data[element]
        else:
            data = self.data

        # Return normalised via calibration
        return (data - self.intercept) / self.gradient

    def getPixelSize(self):
        return self.speed * self.scantime

    def getAspect(self):
        return 1.0 / ((self.speed * self.scantime) / self.spotsize)

    def getExtent(self):
        shape = self.data.shape
        return (0, self.speed * self.scantime * shape[1],
                self.spotsize * shape[0], 0)
