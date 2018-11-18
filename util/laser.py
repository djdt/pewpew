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


class LaserData(object):
    def __init__(self, data, params=LaserParams()):
        self.data = data
        self.params = params

    def isotopes(self):
        return self.data.dtype.names

    def extent(self):
        x = self.data.shape[1] * self.params.pixelsize()[0]
        y = self.data.shape[0] * self.params.pixelsize()[1]
        return (0, x, 0, y)

    def aspect(self):
        return self.params.aspect()


    def get(self, isotope=None):
        if isotope is not None:
            return self.data[isotope]
        else:
            return self.data

# class Laser(object):
#     DEFAULT_PARAMS = {
#         "scantime": 1.0,  # s
#         "speed": 1.0,     # um/s
#         "spotsize": 1.0,  # um
#         "gradient": 1.0,
#         "intercept": 1.0
#     }

#     def __init__(self, scantime=0.25, speed=120.0, spotsize=30.0,
#                  gradient=1.0, intercept=0.0):
#         self.data = []

#         self.scantime = scantime  # s
#         self.speed = speed        # um/s
#         self.spotsize = spotsize  # um

#         self.gradient = gradient
#         self.intercept = intercept

#     def getIsotopes(self):
#         return self.data.dtype.names

#     def importData(self, path, importer='Agilent'):
#         if importer is 'Agilent':
#             self.data = importAgilentBatch(path)
#         else:
#             print(f'Laser.import: unknown importer \'{importer}\'!')

#     def getData(self, element=None):
#         if element is not None:
#             data = self.data[element]
#         else:
#             data = self.data

#         # Return normalised via calibration
#         return (data - self.intercept) / self.gradient

#     def getPixelSize(self):
#         return self.speed * self.scantime

#     def getAspect(self):
#         return 1.0 / ((self.speed * self.scantime) / self.spotsize)

#     def getExtent(self):
#         shape = self.data.shape
#         return (0, self.speed * self.scantime * shape[1],
#                 self.spotsize * shape[0], 0)
