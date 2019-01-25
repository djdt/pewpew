import sys
import numpy as np

from pewpew.lib.laser import LaserData


def save(path: str, laser: LaserData) -> None:
    data = np.reshape(laser.data, (*laser.data.shape, 1))
    nx, ny, nz = data.shape

    extent_str = f"0 {nx-1} 0 {ny-1} 0 {nz-1}"
    endian = "LittleEndian" if sys.byteorder == "little" else "BigEndian"

    extent = laser.extent()
    depth = laser.config["spotsize"]

    coords = [
        np.linspace(extent[2], extent[3], nx),
        np.linspace(extent[0], extent[1], ny),
        np.linspace(0, nz * -depth, nz),
    ]

    offset = 0
    with open(path, "wb") as fp:
        fp.write(
            (
                '<?xml version="1.0"?>\n'
                '<VTKFile type="RectilinearGrid" version="1.0" '
                f'byte_order="{endian}" header_type="UInt64">\n'
                f'<RectilinearGrid WholeExtent="{extent_str}">\n'
                f'<Piece Extent="{extent_str}">\n'
            ).encode()
        )

        fp.write("<Coordinates>\n".encode())
        for i, coord in zip(["x", "y", "z"], coords):
            fp.write(
                (
                    f'<DataArray Name="{i}_coordinates" type="Float64" '
                    f'format="appended" offset="{offset}"/>\n'
                ).encode()
            )
            offset += coord.size * coord.itemsize + 8  # 8 for blocksize
        fp.write("</Coordinates>\n".encode())

        fp.write(f'<PointData Scalars="{data.dtype.names[0]}">\n'.encode())
        for name in data.dtype.names:
            fp.write(
                (
                    f'<DataArray Name="{name}" type="Float64" '
                    f'format="appended" offset="{offset}"/>\n'
                ).encode()
            )
            offset += data[name].size * data[name].itemsize + 8  # blocksize
        fp.write("</PointData>\n".encode())

        fp.write(
            (
                "</Piece>\n"
                "</RectilinearGrid>\n"
                '<AppendedData encoding="raw">\n'
                "_"
            ).encode()
        )

        for coord in coords:
            fp.write(np.uint64(coord.size * coord.itemsize))
            fp.write(coord)

        for name in data.dtype.names:
            fp.write(np.uint64(data[name].size * data[name].itemsize))
            fp.write(data[name].ravel("F"))

        fp.write(("</AppendedData>\n" "</VTKFile>").encode())
