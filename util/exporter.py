import numpy as np
import base64

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from util.laserimage import plotLaserImage


def exportNpz(path, laserdata_list):
    savedict = {'_type': [], '_config': []}
    for i, ld in enumerate(laserdata_list):
        savedict['_type'].append(type(ld))
        savedict['_config'].append(ld.config)
        savedict[f'_data{i}'] = ld.data
    np.savez(path, **savedict)


def exportPng(path, data, isotope, aspect, extent, viewconfig):
    fig = Figure(frameon=False, tight_layout=True,
                 figsize=(5, 5), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    plotLaserImage(fig, ax, data, label=isotope, colorbar='bottom',
                   cmap=viewconfig['cmap'], aspect=aspect, extent=extent,
                   interpolation=viewconfig['interpolation'],
                   vmin=viewconfig['cmap_range'][0],
                   vmax=viewconfig['cmap_range'][1])
    fig.savefig(path, transparent=True, frameon=False)
    fig.clear()
    canvas.close()


def exportVtr(path, data, extent, spotsize):
    nx, ny, nz = data.shape

    header = ("<?xml version=\"1.0\"?>\n"
              "<VTKFile type=\"RectilinearGrid\" version=\"1.0\" "
              "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
              f"<RectilinearGrid WholeExtent=\"0 {nx} 0 {ny} 0 {nz}\">\n"
              f"<Piece Extent=\"0 {nx} 0 {ny} 0 {nz}\">\n")

    x_coords = np.linspace(extent[0], extent[1], nx)
    y_coords = np.linspace(extent[2], extent[3], ny)
    z_coords = np.linspace(0, nz * spotsize, nz)

    with open(path, 'wb') as fp:
        fp.write(header.encode('utf-8'))
        fp.write("<Coordinates>\n".encode('utf-8'))
        for name, coords in zip(
                ['x_coordinates', 'y_coordinates', 'z_coordinates'],
                [x_coords, y_coords, z_coords]):
            fp.write((f"<DataArray Name=\"{name}\" "
                      "type=\"Float64\" format=\"appended\">\n").encode('utf-8')
            fp.write(base64.b64encode(coords)))
            fp.write("</DataArray>\n")
        fp.write("</Coordinates>\n")
        fp.write(f"<PointData scalars=\"Isotopes\">\n")
        for isotope in data.dtype.names:
            fp.write(f"<DataArray Name=\"{isotope}\" "
                     "type=\"Float64\" format=\"binary\"/>\n")
            fp.write(str(base64.b64encode(np.ascontiguousarray(data[isotope]))))
            fp.write("</DataArray>\n")
        fp.write("</PointData>\n</Piece>\n</RectilinearGrid>\n</VTKFile>\n")
