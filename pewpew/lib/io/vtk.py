import sys
import numpy as np

from pewpew.lib.laser import Laser
from typing import Dict
from pewpew.lib.krisskross import KrissKross


# TODO implement extent export


def save(path: str, laser: Laser) -> None:
    """Save data as a VTK ImageData XML."""

    data: Dict[str, np.ndarray] = {}
    for k in laser.isotopes():
        if isinstance(laser, KrissKross):
            data[k] = laser.get(k, calibrate=True, flat=False)
        else:
            data[k] = laser.get(k, calibrate=True)
            data[k] = np.reshape(data[k], (*data[k].shape, 1))

    nx, ny, nz = data[laser.isotopes()[0]].shape
    origin = 0.0, 0.0

    endian = "LittleEndian" if sys.byteorder == "little" else "BigEndian"

    extent_str = f"0 {nx} 0 {ny} 0 {nz}"
    origin_str = f"{origin[1]} {origin[1]} 0.0"
    spacing = *laser.config.pixel_size(), laser.config.spotsize / 2.0
    spacing_str = f"{spacing[0]} {spacing[1]} {spacing[2]}"

    offset = 0
    with open(path, "wb") as fp:
        fp.write(
            (
                '<?xml version="1.0"?>\n'
                '<VTKFile type="ImageData" version="1.0" '
                f'byte_order="{endian}" header_type="UInt64">\n'
                f'<ImageData WholeExtent="{extent_str}" '
                f'Origin="{origin_str}" Spacing="{spacing_str}">\n'
                f'<Piece Extent="{extent_str}">\n'
            ).encode()
        )

        fp.write(f'<CellData Scalars="{laser.isotopes()[0]}">\n'.encode())
        for k, v in data.items():
            fp.write(
                (
                    f'<DataArray Name="{k}" type="Float64" '
                    f'format="appended" offset="{offset}"/>\n'
                ).encode()
            )
            offset += v.size * v.itemsize + 8  # blocksize
        fp.write("</CellData>\n".encode())

        fp.write(
            (
                "</Piece>\n" "</ImageData>\n" '<AppendedData encoding="raw">\n' "_"
            ).encode()
        )

        for k, v in data.items():
            fp.write(np.uint64(v.size * v.itemsize))
            fp.write(v.ravel("F"))

        fp.write(("</AppendedData>\n" "</VTKFile>").encode())
