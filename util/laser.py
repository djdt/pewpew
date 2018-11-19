# from util.importers import importAgilentBatch
import os.path
import numpy as np



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

    def extent(self, data_shape):
        x = data_shape[1] * self.pixelsize()[0]
        y = data_shape[0] * self.pixelsize()[1]
        return (0, x, 0, y)


class LaserData(object):
    def __init__(self, data=None, isotope="", config=LaserConfig(),
                 source=""):
        self.data = data
        self.isotope = isotope
        self.config = config
        self.source = ""

    def open(path):
        pass

    def save(path, laserdata_list):
        savedict = {'isotope': [], 'config': [], 'data': []}
        for ld in laserdata_list:
            savedict['isotope'].append(ld.isotope)
            savedict['config'].append(ld.config)
            savedict['data'].append(ld.data)

        np.savez(path, **savedict)

    def openCsv(self, path):
        self.source = path
        with open(path, 'r') as fp:
            self.isotope = fp.readline().rstrip()
            self.config = LaserConfig()
            sconfig = fp.readline().split(',')
            for sp in sconfig:
                k, v = sp.split('=')
                self.config[k] = v

    def saveCsv(self, path):
        pstring = ','.join(
            [f'{p}={self.config[p]}' for p in LaserConfig.EDITABLE])
        header = f'{self.isotope}\n{pstring}\n'
        np.savetxt(path, delimiter=',', header=header)

