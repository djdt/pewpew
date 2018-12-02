import numpy as np
import struct

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
              f"<RectilinearGrid WholeExtent=\"0 {nx-1} 0 {ny-1} 0 {nz-1}\">\n"
              f"<Piece Extent=\"0 {nx-1} 0 {ny-1} 0 {nz-1}\">\n")

    coords = [np.linspace(extent[2], extent[3], nx),
              np.linspace(extent[0], extent[1], ny),
              np.linspace(0, nz * -spotsize, nz)]

    offset = 0

    coordinates = "<Coordinates>\n"
    for i, coord in zip(['x', 'y', 'z'], coords):
        coordinates += (f"<DataArray Name=\"{i}_coordinates\" type=\"Float64\""
                        f" format=\"appended\" offset=\"{offset}\"/>\n")
        offset += coord.size * coord.itemsize + 8  # 8 for blocksize
    coordinates += "</Coordinates>\n"

    point_data = "<PointData Scalars=\"{data.dtype.names[0]}\">\n"
    for name in data.dtype.names:
        point_data += (f"<DataArray Name=\"{name}\" type=\"Float64\""
                       f" format=\"appended\" offset=\"{offset}\"/>\n")
        offset += data[name].size * data[name].itemsize + 8  # 8 for blocksize
    point_data += "</PointData>\n"

    with open(path, 'wb') as fp:
        fp.write(header.encode())
        fp.write(coordinates.encode())
        fp.write(point_data.encode())
        fp.write(("</Piece>\n</RectilinearGrid>\n"
                  "<AppendedData encoding=\"raw\">\n_").encode())

        for coord in coords:
            fp.write(struct.pack('<Q', coord.size * coord.itemsize))
            binary = struct.pack(f'<{coord.size}d',
                                 *np.ravel(coord, order='F'))
            fp.write(binary)

        for name in data.dtype.names:
            fp.write(struct.pack('<Q', data[name].size * data[name].itemsize))
            binary = struct.pack(f'<{data[name].size}d',
                                 *np.ravel(data[name], order='F'))
            fp.write(binary)

        fp.write("</AppendedData>\n</VTKFile>".encode())
