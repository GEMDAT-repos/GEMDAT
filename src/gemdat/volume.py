"""This module contains functions related to dealing with volumetric data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import scipy.ndimage as ndi
from pymatgen.core import Structure
from pymatgen.core.units import Unit
from pymatgen.io.vasp import VolumetricData
from scipy.constants import physical_constants
from skimage.feature import blob_dog
from skimage.measure import regionprops

from ._plot_backend import plot_backend
from .segmentation import watershed_pbc

if TYPE_CHECKING:
    import networkx as nx
    from pymatgen.core import Lattice, PeriodicSite
    from skimage.measure._regionprops import RegionProperties

    from gemdat.path import Pathway
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
    label : str
        Label for the Volume
    units : Unit | None
        Optional unit for the data
    """

    data: np.ndarray
    lattice: Lattice
    label: str = 'volume'
    units: Unit = field(default_factory=lambda: Unit(''))

    def __post_init__(self):
        self.dims = self.data.shape

    def __repr__(self):
        units = f' ({self.units})' if self.units else ''

        def to_str(x):
            return f'{x:>10.6f}'

        abc = ' '.join(to_str(i) for i in self.lattice.abc)
        angles = ' '.join(to_str(i) for i in self.lattice.angles)
        pbc = ' '.join(str(p).rjust(10) for p in self.lattice.pbc)

        s = (
            f'Data: {self.dims}',
            f'Lattice, abc: {abc}',
            f'      angles: {angles}',
            f'         pbc: {pbc}',
            f'Label: {self.label}{units}',
        )
        return '\n'.join(s)

    def normalized(self) -> np.ndarray:
        """Return normalized data."""
        return self.data / self.data.max()

    def probability(self) -> np.ndarray:
        """Return probability data."""
        return self.data / self.data.sum()

    @property
    def voxel_size(self) -> np.ndarray:
        """Return voxel size in Angstrom."""
        return np.array(self.lattice.lengths) / self.dims

    @classmethod
    def from_volumetric_data(cls, volume: VolumetricData):
        """Create instance from VolumetricData.

        Parameters
        ----------
        volume : pymatgen.io.common.VolumetricData
            Input volumetric data
        """
        return cls(
            data=volume.data['total'],
            lattice=volume.structure.lattice,
        )

    def voxel_to_cart_coords(self, voxel: np.ndarray | list[Any]) -> np.ndarray:
        """Convert voxel coordinates to cartesian coordinates.

        Parameters
        ----------
        voxel : tuple[int, int, int]
            Input voxel coordinates

        Returns
        -------
        np.ndarray
            Output cartesian coordinates
        """
        frac_coords = self.voxel_to_frac_coords(voxel)
        return self.lattice.get_cartesian_coords(frac_coords)

    def voxel_to_frac_coords(self, voxel: np.ndarray | list[Any]) -> np.ndarray:
        """Convert voxel coordinates to fractional coordinates.

        Parameters
        ----------
        voxel : tuple[int, int, int]
            Input voxel coordinates

        Returns
        -------
        np.ndarray
            Output fractional coordinates
        """
        return (np.array(voxel) + 0.5) / np.array(self.dims)

    def frac_coords_to_voxel(self, frac_coords: np.ndarray) -> np.ndarray:
        """Convert fractional coordinates to voxel coordinates.

        Parameters
        ----------
        frac_coords : tuple[int, int, int]
            Input fractional coordinates

        Returns
        -------
        np.ndarray
            Output voxel coordinates
        """
        return (np.array(frac_coords) * np.array(self.dims)).astype(int)

    def site_to_voxel(self, site: PeriodicSite) -> np.ndarray:
        """Convert site coordinates to voxel coordinates.

        Parameters
        ----------
        site : PeriodicSite
            Input site

        Returns
        -------
        np.ndarray
            Output voxel coordinates
        """
        return self.frac_coords_to_voxel(site.frac_coords)

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
            Extend the volume by this number of voxels by wrapping around.
            This helps finding maxima for blobs sitting at the edge of the
            unit cell.
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
        data = self.normalized()
        data = np.pad(data, pad_width=pad, mode='wrap')

        coords = blob_dog(data, **kwargs)[:, 0:3]
        coords = coords - np.array((pad, pad, pad))

        if remove_outside:
            imax, jmax, kmax = self.dims
            imin, jmin, kmin = 0, 0, 0

            c0 = (coords[:, 0] >= imin) & (coords[:, 0] < imax)
            c1 = (coords[:, 1] >= jmin) & (coords[:, 1] < jmax)
            c2 = (coords[:, 2] >= kmin) & (coords[:, 2] < kmax)

            coords = coords[c0 & c1 & c2]

        return coords[:, 0:3].astype(int)

    def to_vasp_volume(
        self,
        structure: Structure,
        *,
        filename: str | None = None,
        other: list[Volume] | None = None,
    ) -> VolumetricData:
        """Convert to vasp volume.

        Parameters
        ----------
        structure : pymatgen.core.structure.Structure
            structure to include in the vasp file (e.g. trajectory structure)
            Also useful if you want to output the density for a select number of
            species, and show the host structure.
        filename : Optional[str]
            If specified, save volume to this filename.
        other : list[Volume]
            Other volumes to store to the vasp volume. Lattice must match to this
            volumes lattice. The volume label is used as the key in the output
            volumetric data.

        Returns
        -------
        vol_vasp : pymatgen.io.vasp.VolumetricData
            Output volume
        """
        data = {'total': self.data}

        if other:
            for volume in other:
                assert volume.lattice == self.lattice
                data[volume.label] = volume.data

        vol_vasp = VolumetricData(
            structure=structure,
            data=data,
        )

        if filename:
            vol_path = Path(filename).with_suffix('.vasp')
            vol_vasp.write_file(vol_path)

        return vol_vasp

    def _peaks_to_props(
        self, peaks: np.ndarray, background_level: float
    ) -> list[RegionProperties]:
        """Segment volume using watershed algorithm.

        Return regionprops.
        """
        data = self.normalized()

        background_level = background_level * data.max()

        mask = np.zeros(data.shape, dtype=bool)
        mask[tuple(peaks.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed_pbc(-data, markers, mask=data > background_level)

        return regionprops(labels, intensity_image=data)

    def _props_to_frac_coords_centroid(
        self,
        *,
        props: list[RegionProperties],
        **kwargs,
    ) -> np.ndarray:
        """Generate fractional coords using centroid method."""
        centroids = []

        for prop in props:
            coords = prop.coords

            for axis in (0, 1, 2):
                dim = self.dims[axis]
                if prop.image.shape[axis] == dim:
                    sel = coords[:, axis] < dim / 2
                    coords[sel, axis] += dim

            centroids.append(coords.mean(axis=0))

        centroids = np.array(centroids)

        # Move coords to voxel center by shifting 0.5
        frac_coords = (centroids + 0.5) / np.array(self.dims)

        return np.array(frac_coords)

    def to_structure(
        self,
        *,
        specie: str = 'X',
        background_level: float = 0.1,
        peaks: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Structure:
        """Converts a volume back to a structure using peak detection. Uses the
        'centroid' method that takes the weighted centroid of all voxels in a
        labeled region (fast),

        Parameters
        ----------
        specie : str
            Specie to assign to the found sites, defaults to 'X'
        background_level : float
            Fraction of the maximum volume value to set as the minimum value
            for peak segmentation.
            Essentially sets `vol_min = background_level * max(vol)`.
            All values below `vol_min` are masked in the peak search.
            Must be between 0 and 1
        peaks : Optional[np.ndarray]
            Voxel coordinates to use as starting points for watershed algorithm.
        **kwargs : dict
            These keywords parameters are passed to [gemdat.Volume.find_peaks][].
            Only applies if `peaks == None`.

        Returns
        -------
        structure : pymatgen.core.structure.Structure
            Output structure
        """
        if peaks is None:
            peaks = self.find_peaks(**kwargs)

        props = self._peaks_to_props(peaks=peaks, background_level=background_level)

        props_to_frac_coords = self._props_to_frac_coords_centroid

        frac_coords = props_to_frac_coords(props=props)

        frac_coords = np.mod(frac_coords, 1)

        structure = Structure(
            lattice=self.lattice,
            coords=frac_coords,
            species=[specie for _ in frac_coords],
        )

        structure.merge_sites(tol=0.1, mode='average')

        return structure

    def get_free_energy(
        self,
        temperature: float,
    ) -> Volume:
        """Estimate the free energy from volume.

        Parameters
        ----------
        temperature : float
            The temperature of the simulation

        Returns
        -------
        free_energy : ndarray
            Free energy in eV on the voxel grid
        """
        prob = self.probability()
        free_energy = (
            -temperature * physical_constants['Boltzmann constant in eV/K'][0] * np.log(prob)
        )

        return FreeEnergyVolume(
            data=np.nan_to_num(free_energy),
            lattice=self.lattice,
        )

    @plot_backend
    def plot_3d(self, *, module, **kwargs):
        """See [gemdat.plots.plot_3d][] for more info."""
        return module.plot_3d(volume=self, **kwargs)


class FreeEnergyVolume(Volume):
    def free_energy_graph(self, **kwargs) -> nx.Graph:
        """Compute the graph of the free energy for networkx functions.

        See [gemdat.path.free_energy_graph][] for more info.
        """
        from .path import free_energy_graph

        return free_energy_graph(self.data, **kwargs)

    def optimal_path(self, F_graph: nx.Graph | None = None, **kwargs) -> Pathway:
        """Calculate the shortest cost-effective path using the desired method.

        Parameters
        ----------
        F_graph : Graph | None
            Optionally, define your own free energy graph. Otherwise,
            it will be calculated on the fly using default parameters.
        **kwargs:
            These parameters are passed to [gemdat.path.optimal_path][].
            See [gemdat.path.optimal_path][] for more info.

        Returns
        -------
        path : Pathway
            Voxel coordinates and energy of optimal path from start to stop.
        """
        from .path import optimal_path

        if not F_graph:
            F_graph = self.free_energy_graph(max_energy_threshold=1e7)

        path = optimal_path(F_graph, **kwargs)
        path.dims = self.dims
        return path

    def optimal_n_paths(self, F_graph: nx.Graph | None = None, **kwargs) -> list[Pathway]:
        """Calculate the n_paths shortest paths between two sites on the graph.

        See [gemdat.path.optimal_n_paths][] for more info.
        """
        from .path import optimal_n_paths

        if not F_graph:
            F_graph = self.free_energy_graph(max_energy_threshold=1e7)

        paths = optimal_n_paths(F_graph, **kwargs)

        for path in paths:
            path.dims = self.dims
        return paths

    def optimal_percolating_path(self, **kwargs) -> Pathway | None:
        """Calculate the optimal percolating path.

        See [gemdat.path.optimal_percolating_path][] for more info.
        """
        from .path import optimal_percolating_path

        return optimal_percolating_path(self, **kwargs)


def trajectory_to_volume(
    trajectory: Trajectory,
    resolution: float = 0.2,
) -> Volume:
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

    digitized_coords = np.vstack(
        [
            np.digitize(coords[:, 0], bins=xbins),
            np.digitize(coords[:, 1], bins=ybins),
            np.digitize(coords[:, 2], bins=zbins),
        ]
    ).T

    indices, counts = np.unique(digitized_coords, return_counts=True, axis=0)
    i, j, k = indices.T

    data = np.zeros((nx - 1, ny - 1, nz - 1), dtype=int)
    data[i, j, k] = counts

    return Volume(
        data=data,
        lattice=lattice,
        label='trajectory',
    )
