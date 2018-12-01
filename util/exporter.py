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


def exportVtr(path, kkdata):
    nx, ny, nz = kkdata.data.shape
    startx, endx, starty, endy = kkdata.extent()

    header = (f"<?xml version=\"1.0\"?>\n"
              "<VTKFile type=\"RectilinearGrid\" version=\"1.0\" "
              "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
              "<RectilinearGrid WholeExtent=\"0 {nx} 0 {ny} 0 {nz}\">\n"
              "<Piece Extent=\"0 {nx} 0 {ny} 0 {nz}\">\n")

    x_coords = np.linspace(startx, endx, kkdata.nx)
    y_coords = np.linspace(starty, endy, kkdata.ny)
    z_coords = np.linspace(0, nz * kkdata.config['spotsize'], nz)

    with open(path, 'wb') as fp:
        fp.write(header)

        fp.write("<Coordinates>\n"
                 "<DataArray Name=\"x_coordinates\" "
                 "type=\"Float64\" format=\"binary\">\n")
        fp.write(base64.b64encode(x_coords))
        fp.write("</DataArray>\n")
        fp.write("<Coordinates>\n"
                 "<DataArray Name=\"x_coordinates\" "
                 "type=\"Float64\" format=\"binary\">\n")
        fp.write(base64.b64encode(y_coords))
        fp.write("</DataArray>\n")
        fp.write("<Coordinates>\n"
                 "<DataArray Name=\"x_coordinates\" "
                 "type=\"Float64\" format=\"binary\">\n")
        fp.write(base64.b64encode(z_coords))
        fp.write("</DataArray>\n")
        fp.write("</Coordinates>\n")

        fp.write(f"<PointData scalars=\"{kkdata.isotopes()[0]}\">\n")
        for isotope in kkdata.isotopes():
            fp.write(f"<DataArray Name=\"{isotope}\" "
                     "type=\"Float64\" format=\"binary\"/>\n")
            fp.write(base64.b64encode(kkdata.calibrated(isotope, flat=False)))
            fp.write("</DataArray>\n")
        fp.write("</PointData>\n</Piece>\n</RectilinearGrid>\n</VTKFile>\n")
