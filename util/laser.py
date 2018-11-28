class LaserData(object):
    DEFAULT_CONFIG = {
        'spotsize': 30.0, 'speed': 120.0, 'scantime': 0.25,
        'gradient': 1.0, 'intercept': 0.0}

    def __init__(self, data=None, config=None,
                 source=""):
        self.data = data
        self.config = LaserData.DEFAULT_CONFIG if config is None else config
        self.source = source

    def get(self, isotope=None):
        if isotope is None:
            return self.data
        return self.data[isotope]

    def calibrated(self, isotope=None):
        return (self.get(isotope) - self.config['intercept']) / \
                self.config['gradient']

    def pixelsize(self):
        return (self.config['speed'] * self.config['scantime'],
                self.config['spotsize'])

    def aspect(self):
        return self.config['spotsize'] / \
               (self.config['speed'] * self.config['scantime'])

    def extent(self):
        # Image data is stored [rows][cols]
        x = self.data.shape[1] * self.pixelsize()[0]
        y = self.data.shape[0] * self.pixelsize()[1]
        return (0, x, 0, y)
