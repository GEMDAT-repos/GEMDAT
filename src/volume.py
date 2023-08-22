from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import VolumetricData

if TYPE_CHECKING:
    from gemdat.trajectory import Trajectory


def trajectory_to_volume(
    trajectory: Trajectory,
    resolution: float = 0.2,
) -> np.ndarray:
    """Calculate density volume from list of coordinates.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    resolution : float, optional
        Minimum resolution for the voxels in Angstrom

    Returns
    -------
    vol : np.ndarray
        3D numpy volume array
    """
    lattice = trajectory.get_lattice()

    coords = trajectory.positions.reshape(-1, 3)

    # coords must be between >= 0, < 1
    assert coords.min() >= 0
    assert coords.max() < 1

    x0 = y0 = z0 = 0
    x1 = y1 = z1 = 1

    nx = int(1 + lattice.lengths[0] // resolution)
    ny = int(1 + lattice.lengths[1] // resolution)
    nz = int(1 + lattice.lengths[2] // resolution)

    # Drop first item, because bins are open-ended on left side
    xbins = np.linspace(x0, x1, nx)[1:]
    ybins = np.linspace(y0, y1, ny)[1:]
    zbins = np.linspace(z0, z1, nz)[1:]

    digitized_coords = np.vstack([
        np.digitize(coords[:, 0], bins=xbins),
        np.digitize(coords[:, 1], bins=ybins),
        np.digitize(coords[:, 2], bins=zbins),
    ]).T

    indices, counts = np.unique(digitized_coords, return_counts=True, axis=0)
    i, j, k = indices.T

    vol = np.zeros((nx - 1, ny - 1, nz - 1), dtype=int)
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


import scipy.ndimage as ndi
from pymatgen.core import Lattice
from pymatgen.io.cif import CifWriter
from skimage import feature
from skimage.measure import regionprops
from skimage.segmentation import watershed


def volume_to_structure(
    vol: VolumetricData | np.ndarray,
    *,
    lattice: Lattice,
    specie: str,
    name: str = 'structure',
    min_n: int = 30,
    **kwargs,
) -> Structure:
    """Converts a volume array back to a structure using peak detection.

    Parameters
    ----------
    vol : VolumetricData | np.ndarray
        Description
    min_n : int, optional
        Description
    **kwargs
        Additional keyword arguments are passed to skimage.feature.blob_dog
    """
    if isinstance(vol, VolumetricData):
        vol = vol.data['total']

    def save(coords, name):
        print('Saving', name, '...')
        s = Structure(lattice=lattice,
                      coords=coords,
                      species=[specie for _ in coords])
        VolumetricData(structure=s, data={
            'total': vol
        }).write_file(name + '.vasp')
        CifWriter(s).write_file(name + '.cif')

    # normalize data
    data = 256 * vol / vol.max()

    coords = feature.blob_dog(data, **kwargs)
    coords = coords[:, 0:3].astype(int)

    # Normalize to volume
    save((coords / np.array(vol.shape)), f'{name}-dog')

    mask = np.zeros(data.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-data, markers, mask=data > min_n)

    props = regionprops(labels, intensity_image=data)
    cw_coords = np.array([p.centroid_weighted for p in props])
    # intensities = np.array([p.intensity_mean for p in props])
    # areas = np.array([p.area_filled for p in props])

    # Move coords to voxel center by shifting 0.5
    frac_coords = (cw_coords + 0.5) / np.array(vol.shape)

    # Normalize to volume
    save(frac_coords, f'{name}-props')

    return Structure(lattice=lattice,
                     coords=frac_coords,
                     species=[specie for _ in frac_coords])
