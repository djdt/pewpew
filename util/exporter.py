import numpy as np


def exportNpz(path, laserdata_list):
    savedict = {'_type': [], '_config': []}
    for i, ld in enumerate(laserdata_list):
        savedict['_type'].append(type(ld))
        savedict['_config'].append(ld.config)
        savedict[f'_data{i}'] = ld.data
    np.savez(path, **savedict)


def exportVtr(path, kkdata, name="laser_plot export"):
    x, y, z = kkdata.data.shape
