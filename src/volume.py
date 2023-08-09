from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from pymatgen.io.vasp import VolumetricData

if TYPE_CHECKING:
    from gemdat.trajectory import Trajectory
    from pymatgen.core import Structure


def trajectory_to_volume(trajectory: Trajectory,
                         resolution: float = 0.2,
                         cartesian: bool = False) -> np.ndarray:
    """Calculate density volume from list of coordinates.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    resolution : float, optional
        Minimum resolution for the voxels in Angstrom
    cartesian : bool, optional
        If True, return volume on a cartesian grid.
        Useful for generic 3D volume viewers

    Returns
    -------
    vol : np.ndarray
        3D numpy volume array
    """
    lattice = trajectory.get_lattice()

    coords = trajectory.coords.reshape(-1, 3)

    if cartesian:
        coords = lattice.get_cartesian_coords(coords)

        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0)

        nx = int(1 + (x1 - x0) // resolution)
        ny = int(1 + (y1 - y0) // resolution)
        nz = int(1 + (z1 - z0) // resolution)
    else:
        x0 = y0 = z0 = 0
        x1 = y1 = z1 = 1

        nx = int(1 + lattice.lengths[0] // resolution)
        ny = int(1 + lattice.lengths[1] // resolution)
        nz = int(1 + lattice.lengths[2] // resolution)

    xbins = np.linspace(x0, x1, nx)
    ybins = np.linspace(y0, y1, ny)
    zbins = np.linspace(z0, z1, nz)

    digitized_coords = np.vstack([
        np.digitize(coords[:, 0], bins=xbins),
        np.digitize(coords[:, 1], bins=ybins),
        np.digitize(coords[:, 2], bins=zbins),
    ]).T

    indices, counts = np.unique(digitized_coords, return_counts=True, axis=0)
    i, j, k = indices.T

    vol = np.zeros((nx + 1, ny + 1, nz + 1), dtype=int)
    vol[i, j, k] = counts

    return vol


def trajectory_to_vasp_volume(trajectory: Trajectory,
                              structure: Optional[Structure] = None,
                              resolution: float = 0.2,
                              filename: str | None = None) -> VolumetricData:
    """Calculate density volume as from list of coordinates.

    Parameters
    ----------
    trajectory : np.ndarray
        Input trajectory
    structure : Optional[Structure]
        structure to include in the vasp file, defaults to trajectory structure.
        Useful if you want to output the density for a select number of species,
        and show the host structure. Defaults to first structure in
        given trajectory (base coordinates).
    resolution : float, optional
        Minimum resolution for the voxels in Angstrom
    filename : str | None, optional
        If specified, save volume to this filename.

    Returns
    -------
    vol : VolumetricData
        Output volumetric data object
    """
    vol = trajectory_to_volume(trajectory=trajectory, resolution=resolution)

    structure = structure if structure else trajectory.get_structure(0)

    vasp_vol = VolumetricData(structure=structure, data={'total': vol})

    if filename:
        vasp_vol.write_file(filename)

    return vasp_vol
