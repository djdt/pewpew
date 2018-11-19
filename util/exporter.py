import numpy as np

from util.laser import LaserData, LaserConfig


def exportNpz(path, laserdata_list):
    savedict = {'isotopes': [], 'configs': []}
    for ld in laserdata_list:
        savedict['isotopes'].append(ld.isotope)
        savedict['configs'].append(ld.config)
        savedict[ld.isotope] = ld.data
    np.savez(path, **savedict)


def exportCsv(path, laserdata):
    pstring = ','.join(
        [f'{p}={laserdata.config[p]}' for p in LaserConfig.EDITABLE])
    header = f'{laserdata.isotope}\n{pstring}\n'
    np.savetxt(path, laserdata.data, delimiter=',', header=header)
