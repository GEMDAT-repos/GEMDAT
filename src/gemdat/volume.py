from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.ndimage as ndi
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import VolumetricData
from skimage.feature import blob_dog
from skimage.measure import regionprops
from skimage.segmentation import watershed

if TYPE_CHECKING:
    from pymatgen.core import Lattice

    from gemdat.trajectory import Trajectory


@dataclass
class Volume:
    data: np.ndarray
    lattice: Lattice
    # Minimum resolution in Angstrom
    resolution: float | None = None

    @property
    def normalized_data(self):
        return self.data / self.data.max()

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """Return voxel size in Angstrom."""
        return tuple(a / b for a, b in zip(self.lattice.lengths,
                                           self.data.shape))  # type: ignore

    @property
    def nx(self):
        return self.data.shape[0]

    @property
    def ny(self):
        return self.data.shape[1]

    @property
    def nz(self):
        return self.data.shape[2]

    @classmethod
    def from_volumetric_data(cls, vol: VolumetricData):
        return cls(data=vol.data['total'],
                   lattice=vol.structure.lattice,
                   resolution=None)


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

    return Volume(data=vol, resolution=resolution, lattice=lattice)


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


def volume_to_structure(
    vol: Volume | VolumetricData,
    *,
    specie: str,
    pad: int = 3,
    background_level: float = 0.1,
    cif_filename: str | None = None,
    vol_filename: str | None = None,
    **kwargs,
) -> Structure:
    """Converts a volume array back to a structure using peak detection.

    Parameters
    ----------
    vol : Volume | VolumetricData
        Input volume data
    specie : str
        Specie to assign to the found sites
    pad : int
        Extend the volume by this number of voxels by wrapping around. This helps with
        densities at the edge of the volume bounds. Set this value higher for high resolution
        densities.
    background_level : float
        Fraction of the maximum volume value to set as the minimum value for peak finding.
        Essentially sets `vol_min = background_level * max(vol)`.
        All values below `vol_min` are masked in the peak search.
        Must be between 0 and 1
    cif_filename : str
        If specified, write structure in CIF format to this filename
    vol_filename : str
        If specified, write structure and volume in VASP format to this
        filename
    **kwargs
        Additional keyword arguments are passed to skimage.feature.blob_dog
    """
    if isinstance(vol, VolumetricData):
        vol = Volume.from_volumetric_data(vol)

    kwargs.setdefault('threshold', 0.01)

    # normalize data
    data = vol.normalized_data
    data = np.pad(data, pad_width=pad, mode='wrap')

    coords = blob_dog(data, **kwargs)
    coords = coords[:, 0:3].astype(int)

    mask = np.zeros(data.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-data, markers, mask=data > background_level)

    props = regionprops(labels, intensity_image=data)
    centroids = np.array([p.centroid_weighted for p in props])

    # Cut centroids in padded area
    imax, jmax, kmax = pad + np.array(data.shape)
    imin, jmin, kmin = (pad, pad, pad)

    c0 = (centroids[:, 0] >= imin) & (centroids[:, 0] < imax)
    c1 = (centroids[:, 1] >= jmin) & (centroids[:, 1] < jmax)
    c2 = (centroids[:, 2] >= kmin) & (centroids[:, 2] < kmax)

    centroids = centroids[c0 & c1 & c2] - [imin, jmin, kmin]

    # Move coords to voxel center by shifting 0.5
    frac_coords = (centroids + 0.5) / np.array(vol.data.shape)

    # mod to unit cell
    frac_coords = np.mod(frac_coords, 1)

    structure = Structure(lattice=vol.lattice,
                          coords=frac_coords,
                          species=[specie for _ in frac_coords])

    if cif_filename:
        cif_path = Path(cif_filename)
        CifWriter(structure).write_file(cif_path.with_suffix('.cif'))

    if vol_filename:
        vol_path = Path(vol_filename).with_suffix('.vasp')
        VolumetricData(structure=structure, data={
            'total': vol.data
        }).write_file(vol_path)

    return structure
