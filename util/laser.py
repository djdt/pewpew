class LaserConfig(object):
    EDITABLE = ['spotsize', 'speed', 'scantime', 'gradient',
                'intercept']

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

    def extent(self, shape):
        x = shape[1] * self.pixelsize()[0]
        y = shape[0] * self.pixelsize()[1]
        return (0, x, 0, y)


class LaserData(object):
    def __init__(self, data=None, isotope="", config=LaserConfig(),
                 source=""):
        self.data = data
        self.isotope = isotope
        self.config = config
        self.source = source

    def calibrated(self):
        return (self.data - self.config.intercept) / self.config.gradient

    def aspect(self):
        return self.config.aspect()

    def extent(self):
        return self.config.extent(self.data.shape)
