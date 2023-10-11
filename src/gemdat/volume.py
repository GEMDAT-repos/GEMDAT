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

from .segmentation import watershed_pbc

if TYPE_CHECKING:
    from pymatgen.core import Lattice
    from skimage.measure._regionprops import RegionProperties

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
    positions : optional[np.ndarray]
        Input trajectory coordinates
    voxel_mapping : optional[np.ndarray]
        Integer array that maps `positions` onto
        flattened voxel indices
    """
    data: np.ndarray
    lattice: Lattice
    resolution: float | None = None
    positions: np.ndarray | None = None
    voxel_mapping: np.ndarray | None = None

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

    def _peaks_to_props(self, peaks: np.ndarray,
                        background_level: float) -> list[RegionProperties]:
        """Segment volume using watershed algorithm.

        Return regionprops.
        """
        data = self.normalized_data

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
                dim = self.data.shape[axis]
                if prop.image.shape[axis] == dim:
                    sel = (coords[:, axis] < dim / 2)
                    coords[sel, axis] += dim

            centroids.append(coords.mean(axis=0))

        centroids = np.array(centroids)

        # Move coords to voxel center by shifting 0.5
        frac_coords = (centroids + 0.5) / np.array(self.data.shape)

        return np.array(frac_coords)

    def _props_to_frac_coords_cluster(
        self,
        *,
        props: list[RegionProperties],
        **kwargs,
    ) -> np.ndarray:
        """Generate fractional coords using cluster method."""
        voxel_mapping = self.voxel_mapping
        positions = self.positions

        if (voxel_mapping is None) or (positions is None):
            raise ValueError(
                '`self.voxel_mapping` and `self.positions` must be defined.')

        frac_coords = []

        tol = 0.95

        for prop in track(props, transient=True):
            prop_coords_idx = np.ravel_multi_index(prop.coords.T,
                                                   dims=self.data.shape,
                                                   mode='wrap')

            prop_pos_idx = np.isin(voxel_mapping, prop_coords_idx)
            prop_pos = positions[prop_pos_idx]

            extent = prop_pos.max(axis=0) - prop_pos.min(axis=0)

            for axis in (0, 1, 2):
                if extent[axis] > tol:
                    sel = (prop_pos[:, axis] < 0.5)
                    prop_pos[sel, axis] += 1

            frac_coord = prop_pos.mean(axis=0)
            frac_coords.append(frac_coord)

        return np.array(frac_coords)

    def to_structure(
        self,
        *,
        specie: str = 'X',
        background_level: float = 0.1,
        method: str = 'centroid',
        peaks: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Structure:
        """Converts a volume back to a structure using peak detection.

        Parameters
        ----------
        specie : str
            Specie to assign to the found sites, defaults to 'X'
        background_level : float
            Fraction of the maximum volume value to set as the minimum value for peak segmentation.
            Essentially sets `vol_min = background_level * max(vol)`.
            All values below `vol_min` are masked in the peak search.
            Must be between 0 and 1
        method : str
            Select method to use for calculating fractional coordinates.
            'centroid' takes the weighted centroid of all voxels in a labeled region (fast),
            'cluster' takes the mean of all trajectory coordinates that contributed to
            that voxel (slow, but more accurate).
        peaks : Optional[np.ndarray]
            Voxel coordinates to use as starting points for watershed algorithm.
        **kwargs : dict
            These keywords parameters are passed to [Volume.find_peaks][].
            Only applies if `peaks == None`.

        Returns
        -------
        structure : pymatgen.core.structure.Structure
            Output structure
        """
        if peaks is None:
            peaks = self.find_peaks(**kwargs)

        props = self._peaks_to_props(peaks=peaks,
                                     background_level=background_level)

        props_to_frac_coords = {
            'centroid': self._props_to_frac_coords_centroid,
            'cluster': self._props_to_frac_coords_cluster,
        }[method]

        frac_coords = props_to_frac_coords(props=props)

        frac_coords = np.mod(frac_coords, 1)

        structure = Structure(lattice=self.lattice,
                              coords=frac_coords,
                              species=[specie for _ in frac_coords])

        structure.merge_sites(tol=0.1, mode='average')

        return structure


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

    digitized_coords = np.vstack([
        np.digitize(coords[:, 0], bins=xbins),
        np.digitize(coords[:, 1], bins=ybins),
        np.digitize(coords[:, 2], bins=zbins),
    ]).T

    indices, counts = np.unique(digitized_coords, return_counts=True, axis=0)
    i, j, k = indices.T

    data = np.zeros((nx - 1, ny - 1, nz - 1), dtype=int)
    data[i, j, k] = counts
    voxel_mapping = np.ravel_multi_index(tuple(digitized_coords.T),
                                         tuple(data.shape))
    voxel_mapping = voxel_mapping.reshape(len(trajectory),
                                          len(trajectory.species))

    return Volume(
        data=data,
        resolution=resolution,
        lattice=lattice,
        # find better place to store these
        positions=trajectory.positions,
        voxel_mapping=voxel_mapping,
    )

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter
from skimage import measure

def plot_density(lattice, diff_positions, site_positions, site_labels, resolution = 0.1, save_plot = True):
    """Returns 3D plot of probability distributing of diffusing atom.

    Parameters
    ----------
    lattice : pymatgen.core.lattice.Lattice
        Lattice parameters for the volume
    diff_positions : np.ndarray
        Input positions of diffusing atom as 3D numpy array
    site_positions : np.ndarray
        Input positions of sites as 2D numpy array
    site_labels : np.ndarray
        Input labels of sites as 1D numpy array
    resolution : float
        The resolution in Angstrom
    save_plot : bool
        The option to save the plot as .html object
    """

    num_dimensions, num_atoms, num_time_steps = diff_positions.shape

    # Find minimal and maximal x, y, z values in cart_pos
    min_coor = np.sum(lattice * (lattice < 0), axis=0)
    max_coor = np.sum(lattice * (lattice > 0), axis=0)
    length = np.ceil((max_coor - min_coor) / resolution).astype(int)

    # Create the lattice points
    origin = [-min_coor[0], -min_coor[1], -min_coor[2]]
    lat = lattice / resolution
    origin = np.array(origin) / resolution
    lat_orig = lat + origin

    # Initialize the figure
    fig = go.Figure()

    # Plot the lattice vectors
    for i in range(3):
        fig.add_trace(go.Scatter3d(
        x=[origin[0], lat_orig[i, 0]],
        y=[origin[1], lat_orig[i, 1]],
        z=[origin[2], lat_orig[i, 2]],
        mode='lines',
        line=dict(color='black')
        ))

        for j in range(3):
            if i != j:
                fig.add_trace(go.Scatter3d(
                x=[lat_orig[i, 0], lat_orig[i, 0] + lat[j, 0]],
                y=[lat_orig[i, 1], lat_orig[i, 1] + lat[j, 1]],
                z=[lat_orig[i, 2], lat_orig[i, 2] + lat[j, 2]],
                mode='lines',
                line=dict(color='black')
                ))

                for k in range(3):
                    if k != i and k != j:
                        fig.add_trace(go.Scatter3d(
                        x=[lat_orig[i, 0] + lat[j, 0], origin[0] + lat[i, 0] + lat[j, 0] + lat[k, 0]],
                        y=[lat_orig[i, 1] + lat[j, 1], origin[1] + lat[i, 1] + lat[j, 1] + lat[k, 1]],
                        z=[lat_orig[i, 2] + lat[j, 2], origin[2] + lat[i, 2] + lat[j, 2] + lat[k, 2]],
                        mode='lines',
                        line=dict(color='black')
                        ))

    # Create a density box and fill it
    box = np.zeros((length[0], length[1], length[2]))
    pos = np.zeros(3, dtype=int)

    for atom in range(num_atoms):
        for time in range(num_time_steps):
            for i in range(3):
                pos[i] = int((diff_positions[i, atom, time] - min_coor[i]) / resolution)
                if pos[i] < 0:
                    pos[i] = length[i] + pos[i]
            box[pos[0], pos[1], pos[2]] += 1

    # Smooth the density
    box_smooth = gaussian_filter(box, sigma=1.0)

    # Define the colors, isosurface values, and alpha
    colors = ['red', 'yellow', 'cyan']
    iso = [0.25, 0.10, 0.007]  # Adjust isosurface values as needed
    alpha = [0.7, 0.5, 0.35]  # Adjust transparency as needed

    # Plot isosurfaces
    for i in range(len(iso)):
        isoval = iso[i] * np.max(box_smooth)
        verts, faces, _, _ = measure.marching_cubes(box_smooth, level=isoval)
        fig.add_trace(go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=alpha[i],
        color=colors[i],
        showlegend=False
        ))

    # Plot site positions
    for i in range(site_positions.shape[1]):
        point_size = 5
        color = px.colors.qualitative.G10[i]

        fig.add_trace(go.Scatter3d(
        x=[origin[0] + site_positions[0, i] / resolution],
        y=[origin[1] + site_positions[1, i] / resolution],
        z=[origin[2] + site_positions[2, i] / resolution],
        mode='markers',
        name=site_labels[i],
        marker=dict(size=point_size, color=color, line=dict(width=2.5)),
        showlegend=False
        ))

    # Customize the plot layout
    fig.update_layout(
    title='Density of diffusing element',
    scene=dict(aspectmode="manual", aspectratio = dict(x=2, y=1, z=1), xaxis_title='X (Angstrom)', yaxis_title='Y (Angstrom)', zaxis_title='Z (Angstrom)'),
    legend=dict(orientation="h", yanchor="bottom", xanchor="left", x=0, y=-0.1),
    showlegend=True,
    margin=dict(l=0, r=0, b=0, t=0),
    scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=-1, y=1, z=-0.6))
    )
    for trace in fig['data']:
        trace['showlegend'] = False
    if save_plot:
        fig.write_html('Density_plot.html')

    return fig
