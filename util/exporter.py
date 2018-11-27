import numpy as np


def exportNpz(path, laserdata_list):
    savedict = {'datatypes': [], 'isotopes': [], 'configs': []}
    for ld in laserdata_list:
        savedict['datatypes'].append(type(ld))
        savedict['isotopes'].append(ld.isotope)
        savedict['configs'].append(ld.config)
        savedict[ld.isotope] = ld.data
    np.savez(path, **savedict)

def exportVtr(path, kkdata, name="laser_plot export"):
    x, y, z = kkdata.data.shape
