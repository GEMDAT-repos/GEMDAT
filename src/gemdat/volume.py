"""This module contains functions related to dealing with volumetric data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.ndimage as ndi
from pymatgen.core import Structure
from pymatgen.io.vasp import VolumetricData
from rich.progress import track
from skimage.feature import blob_dog
from skimage.measure import regionprops
from skimage.segmentation import watershed

from .segmentation import watershed_pbc

if TYPE_CHECKING:
    from pymatgen.core import Lattice

    from gemdat.trajectory import Trajectory


@dataclass
class Volume:
    """Container for volumetric data.

    Parameters
    ----------
    data : np.ndarray
        Input volume as 3D numpy array
    lattice : pymatgen.core.lattice.Lattice
        Lattice parameters for the volume
    resolution : optional[float]
        The minimum resolution in Angstrom that the volume
        was generated at.
    """
    data: np.ndarray
    lattice: Lattice
    resolution: float | None = None

    @property
    def normalized_data(self) -> np.ndarray:
        """Return normalized data."""
        return self.data / self.data.max()

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """Return voxel size in Angstrom."""
        return tuple(a / b for a, b in zip(self.lattice.lengths,
                                           self.data.shape))  # type: ignore

    @classmethod
    def from_volumetric_data(cls, vol: VolumetricData):
        """Create instance from VolumetricData.

        Parameters
        ----------
        vol : pymatgen.io.common.VolumetricData
            Input volumetric data
        """
        return cls(data=vol.data['total'],
                   lattice=vol.structure.lattice,
                   resolution=None)

    def find_peaks(
        self,
        pad: int = 3,
        remove_outside: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Find peaks using the [Difference of
        Gaussian][skimage.feature.blob_dog] function in [scikit-
        image][skimage].

        Volume data are normalized to (0-1) prior to peak finding.

        Parameters
        ----------
        pad : int
            Extend the volume by this number of voxels by wrapping around. This finding
            maxima for blobs sitting at the edge of the unit cell.
        remove_outside : bool
            If True, remove peaks outside the lattice. Only applicable
            if pad > 0.
        **kwargs
            Additional keyword arguments are passed to [skimage.feature.blob_dog][]

        Returns
        -------
        coords : np.ndarray
            List of coordinates
        """
        kwargs.setdefault('threshold', 0.01)

        # normalize data
        data = self.normalized_data
        data = np.pad(data, pad_width=pad, mode='wrap')

        coords = blob_dog(data, **kwargs)[:, 0:3]
        coords = coords - np.array((pad, pad, pad))

        if remove_outside:
            imax, jmax, kmax = self.data.shape
            imin, jmin, kmin = 0, 0, 0

            c0 = (coords[:, 0] >= imin) & (coords[:, 0] < imax)
            c1 = (coords[:, 1] >= jmin) & (coords[:, 1] < jmax)
            c2 = (coords[:, 2] >= kmin) & (coords[:, 2] < kmax)

            coords = coords[c0 & c1 & c2]

        return coords[:, 0:3].astype(int)

    def to_vasp_volume(self, structure: Structure, *,
                       filename: Optional[str]) -> VolumetricData:
        """Convert to vasp volume.

        Parameters
        ----------
        structure : pymatgen.core.structure.Structure
            structure to include in the vasp file (e.g. trajectory structure)
            Also useful if you want to output the density for a select number of species,
            and show the host structure.
        filename : Optional[str]
            If specified, save volume to this filename.

        Returns
        -------
        vol_vasp : pymatgen.io.vasp.VolumetricData
            Output volume
        """
        if filename:
            vol_path = Path(filename).with_suffix('.vasp')
            vol_vasp = VolumetricData(structure=structure,
                                      data={
                                          'total': self.data
                                      }).write_file(vol_path)
        return vol_vasp

    def to_structure(
        self,
        *,
        specie: str = 'X',
        pad: int = 3,
        background_level: float = 0.1,
        **kwargs,
    ) -> Structure:
        """Converts a volume back to a structure using peak detection.

        Parameters
        ----------
        specie : str
            Specie to assign to the found sites, defaults to 'X'
        pad : int
            Extend the volume by this number of voxels by wrapping around. This helps with
            densities at the edge of the volume bounds. Set this value higher for high resolution
            densities.
        background_level : float
            Fraction of the maximum volume value to set as the minimum value for peak finding.
            Essentially sets `vol_min = background_level * max(vol)`.
            All values below `vol_min` are masked in the peak search.
            Must be between 0 and 1
        **kwargs : dict
            These keywords parameters are passed to [Volume.find_peaks][].

        Returns
        -------
        structure : pymatgen.core.structure.Structure
            Output structure
        """
        data = self.normalized_data
        data = np.pad(data, pad_width=pad, mode='wrap')

        coords = self.find_peaks(pad=pad, **kwargs)
        coords += np.array((pad, pad, pad))

        background_level = background_level * data.max()

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
        frac_coords = (centroids + 0.5) / np.array(self.data.shape)

        # mod to unit cell
        frac_coords = np.mod(frac_coords, 1)

        structure = Structure(lattice=self.lattice,
                              coords=frac_coords,
                              species=[specie for _ in frac_coords])

        structure.merge_sites(tol=0.1, mode='average')

        return structure

    def to_structure2(
        self,
        trajectory: Trajectory,
        *,
        specie: str = 'X',
        background_level: float = 0.1,
        **kwargs,
    ) -> Structure:
        """Converts a volume back to a structure using peak detection.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory for positions and voxel indices
        specie : str
            Specie to assign to the found sites, defaults to 'X'
        background_level : float
            Fraction of the maximum volume value to set as the minimum value for peak finding.
            Essentially sets `vol_min = background_level * max(vol)`.
            All values below `vol_min` are masked in the peak search.
            Must be between 0 and 1
        **kwargs : dict
            These keywords parameters are passed to [Volume.find_peaks][].

        Returns
        -------
        structure : pymatgen.core.structure.Structure
            Output structure
        """
        data = self.normalized_data

        coords = self.find_peaks(**kwargs)

        background_level = background_level * data.max()

        mask = np.zeros(data.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed_pbc(-data, markers, mask=data > background_level)

        props = regionprops(labels, intensity_image=data)

        voxel_coords = trajectory.voxel_index
        positions = trajectory.positions
        frac_coords = []

        tol = 0.95

        for prop in track(props, transient=True):
            prop_coords_idx = np.ravel_multi_index(prop.coords.T,
                                                   dims=self.data.shape,
                                                   mode='wrap')

            prop_pos_idx = np.isin(voxel_coords, prop_coords_idx)
            prop_pos = positions[prop_pos_idx]

            extent = prop_pos.max(axis=0) - prop_pos.min(axis=0)

            for axis in (0, 1, 2):
                if extent[axis] > tol:
                    sel = (prop_pos[:, axis] < 0.5)
                    prop_pos[sel, axis] += 1

            frac_coord = prop_pos.mean(axis=0)
            frac_coords.append(frac_coord)

        frac_coords = np.mod(frac_coords, 1)

        structure = Structure(lattice=self.lattice,
                              coords=frac_coords,
                              species=[specie for _ in frac_coords])

        structure.merge_sites(tol=0.1, mode='average')

        return structure


def trajectory_to_volume(
    trajectory: Trajectory,
    resolution: float = 0.2,
) -> np.ndarray:
    """Calculate density volume from a trajectory.

    All coordinates are binned into voxels. The value of each
    voxel represents the number of coodinates that are associated
    with it.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    resolution : float, optional
        Minimum resolution for the voxels in Angstrom

    Returns
    -------
    vol : Volume
        Output volume
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

    data = np.zeros((nx - 1, ny - 1, nz - 1), dtype=int)
    data[i, j, k] = counts

    voxel_index = np.ravel_multi_index(digitized_coords.T, data.shape)
    voxel_index = voxel_index.reshape(len(trajectory), len(trajectory.species))

    # Find better place to store this
    trajectory.voxel_index = voxel_index

    return Volume(data=data, resolution=resolution, lattice=lattice)
