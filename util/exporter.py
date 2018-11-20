import numpy as np


def exportNpz(path, laserdata_list):
    savedict = {'isotopes': [], 'configs': []}
    for ld in laserdata_list:
        savedict['isotopes'].append(ld.isotope)
        savedict['configs'].append(ld.config)
        savedict[ld.isotope] = ld.data
    np.savez(path, **savedict)


def exportCsv(path, laserdata):
    header = f'{laserdata.isotope}\n{str(laserdata.config)}\n'
    np.savetxt(path, laserdata.data, delimiter=',', header=header)
