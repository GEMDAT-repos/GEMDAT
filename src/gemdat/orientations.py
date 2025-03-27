from __future__ import annotations

from dataclasses import InitVar, dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.symmetry.groups import PointGroup

from gemdat.utils import cartesian_to_spherical, fft_autocorrelation

from ._plot_backend import plot_backend

if TYPE_CHECKING:
    from gemdat.trajectory import Trajectory


@dataclass
class Orientations:
    """Container for orientational data. It computes trajectories of unit
    vectors defined as the distance between a central and satellite atoms,
    meant to track orientation of molecules or clusters.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    center_type: str
        Type of the central atoms
    satellite_type: str
        Type of the satellite atoms
    vectors: np.ndarray
        Vectors representing orientation direction
    """

    trajectory: Trajectory
    center_type: str
    satellite_type: str
    vectors: np.ndarray = field(init=False)
    in_vectors: InitVar[np.ndarray | None] = None

    def __post_init__(self, in_vectors: np.ndarray | None = None):
        """Computes trajectories of unit vectors defined as the distance
        between a central and satellite atoms, meant to track orientation of
        molecules or clusters.

        Parameters
        ----------
        in_vectors : np.ndarray
            Bypass creation of vectors and use these instead.
        """
        if in_vectors is not None:
            self.vectors = in_vectors
        else:
            lattice = self.trajectory.get_lattice()
            direction = self._fractional_directions(self._distances)
            self.vectors = lattice.get_cartesian_coords(direction)

    @property
    def _time_step(self) -> float:
        """Return the time step of the trajectory."""
        assert self.trajectory.time_step
        return self.trajectory.time_step

    @property
    def _trajectory_cent(self) -> Trajectory:
        """Return trajectory of center atoms."""
        return self.trajectory.filter(self.center_type)

    @property
    def _trajectory_sat(self) -> Trajectory:
        """Return trajectory of satellite atoms."""
        return self.trajectory.filter(self.satellite_type)

    @property
    def _distances(self) -> np.ndarray:
        """Calculate distances between every central atom and all satellite
        atoms."""
        central_start_coord = self._trajectory_cent.base_positions
        assert central_start_coord is not None
        satellite_start_coord = self._trajectory_sat.base_positions
        assert satellite_start_coord is not None
        lattice = self.trajectory.get_lattice()
        distance = np.array(
            [
                [
                    lattice.get_all_distances(central, satellite)
                    for satellite in satellite_start_coord
                ]
                for central in central_start_coord
            ]
        )
        return distance

    def _matching_matrix(self, distance: np.ndarray, frac_coord_cent: np.ndarray) -> np.ndarray:
        """Determine which satellite atoms are close enough to central atom to
        be connected.

        Parameters
        ----------
        distance: np.ndarray
            Distance between all the central-satellite pairs
        frac_coord_cent; np.ndarray
            Fractional coordinates of all the central atoms

        Returns
        -------
        matching_matrix: np.ndarray
            Matrix that shows which center-satellite pair is closer than
            the matching criteria
        """
        match_criteria = 1.5 * np.min(distance)

        distance_match = np.where(distance < match_criteria, distance, 0)
        matching_matrix = np.zeros((len(frac_coord_cent[0, :, 0]), 4), dtype=int)
        for k in range(len(frac_coord_cent[0, :, 0])):
            matching_matrix[k, :] = np.where(distance_match[k, :] != 0)[0][:4]
        return matching_matrix

    def _central_satellite_matrix(
        self, distance: np.ndarray, frac_coord_cent: np.ndarray
    ) -> np.ndarray:
        """Get the combinations of central atoms and satellite atoms in a
        matrix.

        Parameters
        ----------
        distance: np.ndarray
            Distance between all the central-satellite pairs
        frac_coord_cent: np.ndarray
            Fractional coordinates of all the central atoms

        Returns
        -------
        combinations: np.ndarray
            Matrix of combinations between central and satellite atoms
        """
        nr_central_atoms = frac_coord_cent.shape[1]

        index_central_atoms = np.arange(nr_central_atoms)

        # index_central_atoms = np.arange(self.nr_central_atoms)
        matching_matrix = self._matching_matrix(distance, frac_coord_cent)
        combinations = np.array(
            [(i, j) for i in index_central_atoms for j in matching_matrix[i, :]]
        )
        return combinations

    def _fractional_directions(self, distance: np.ndarray) -> np.ndarray:
        """Get all fractional coordinate vectors going from central atom to its
        ligands.

        Parameters
        ----------
        distance: np.ndarray
            Distance between all the central-satellite pairs

        Returns
        -------
        direction: np.ndarray
            Contains the direction between central atoms and their ligands.
        """
        frac_coord_cent = self._trajectory_cent.positions
        frac_coord_sat = self._trajectory_sat.positions

        combinations = self._central_satellite_matrix(distance, frac_coord_cent)

        sat = frac_coord_sat[:, combinations[:, 1], :]
        cent = frac_coord_cent[:, combinations[:, 0], :]

        direction = sat - cent

        # Take the periodic boundary conditions into account.
        direction = np.where(direction > 0.5, direction - 1, direction)
        direction = np.where(direction < -0.5, direction + 1, direction)

        return direction

    def normalize(self) -> Orientations:
        """Normalize the trajectory of unit vectors.

        Returns
        -------
        Orientations
        """
        vectors = self.vectors / np.linalg.norm(self.vectors, axis=-1, keepdims=True)

        return replace(self, in_vectors=vectors)

    def symmetrize(
        self, sym_group: str | None = None, sym_ops: np.ndarray | None = None
    ) -> Orientations:
        """Apply symmetry elements to the trajectory to improve statistics.

        One of `sym_group` and `sym_ops` must be supplied.

        Parameters
        ----------
        sym_group: str
            Name of the symmetry group in Hermann-Mauguin notation
        sym_ops: np.ndarray
            Matrix of symmetry operations, overrides `sym_group`

        Returns
        -------
        Orientations
        """
        if sym_ops is not None:
            if sym_ops.ndim != 3:
                # reshape if a single transformation is provided
                sym_ops = sym_ops.reshape(1, 3, 3)
            sym_ops = sym_ops
        elif sym_group:
            g = PointGroup(sym_group)
            sym_ops = np.array(
                [element.rotation_matrix for element in g.symmetry_ops]
            ).transpose(1, 2, 0)
        else:
            raise ValueError('At least one of `sym_group` or `sym_ops` must be provided')

        n_ts = self.vectors.shape[0]
        n_bonds = self.vectors.shape[1]
        n_symops = sym_ops.shape[2]

        direction_sym = np.einsum('tbi,ijk->tbkj', self.vectors, sym_ops)
        vectors = direction_sym.reshape(n_ts, n_bonds * n_symops, 3)

        return replace(self, in_vectors=vectors)

    def transform(self, matrix: np.ndarray) -> Orientations:
        """Convert the trajectory of unit vectors e.g. from primitive to
        conventional setting.

        A conventional unit cell only contains one lattice point, while the
        primitive cell contains the Bravais lattice. This means that the
        conventional form is simpler to visualize and compare.

        This transformation requires the definition of the conversion matrix.

        Parameters
        ----------
        matrix: np.array
            Matrix for vector transformation

        Returns
        -------
        orientations : Orientations
        """
        if matrix.shape != (3, 3):
            raise ValueError('matrix must be a 3x3 matrix')

        vectors = np.dot(self.vectors, matrix.T)

        return replace(self, in_vectors=vectors)

    @property
    def vectors_spherical(self) -> np.ndarray:
        """Return vectors in spherical coordinates in degrees.

        Returns
        -------
        np.array
            azimuth, elevation, length
        """
        return cartesian_to_spherical(self.vectors)

    def autocorrelation(self):
        """Compute the autocorrelation of the orientation vectors using FFT."""
        return fft_autocorrelation(self.vectors)

    @plot_backend
    def plot_rectilinear(self, *, module, **kwargs):
        """See [gemdat.plots.rectilinear][] for more info."""
        return module.rectilinear(orientations=self, **kwargs)

    @plot_backend
    def plot_polar(self, *, module, **kwargs):
        """See [gemdat.plots.polar][] for more info."""
        return module.polar(orientations=self, **kwargs)

    @plot_backend
    def plot_bond_length_distribution(self, *, module, **kwargs):
        """See [gemdat.plots.bond_length_distribution][] for more info."""
        return module.bond_length_distribution(orientations=self, **kwargs)

    @plot_backend
    def plot_autocorrelation(self, *, module, **kwargs):
        """See [gemdat.plots.unit_vector_autocorrelation][] for more info."""
        return module.autocorrelation(orientations=self, **kwargs)


def calculate_spherical_areas(shape: tuple[int, int], radius: float = 1) -> np.ndarray:
    """Calculate the areas of a section of a sphere, defined in spherical
    coordinates. Useful for normalization purposes.

    Parameters
    ----------
    shape : tuple
        Shape of the grid in integer degrees
    radius : float
        Radius of the sphere

    Returns
    -------
    areas : np.ndarray
        Areas of the section
    """
    elevation_angles = np.linspace(0, 180, shape[0])

    areas = np.zeros(shape, dtype=float)
    azimuthal_increment = np.deg2rad(1)
    elevation_increment = np.deg2rad(1)

    for i in range(shape[1]):
        for j in range(shape[0]):
            areas[j, i] = (
                (radius**2)
                * azimuthal_increment
                * np.sin(np.deg2rad(elevation_angles[j]))
                * elevation_increment
            )
            # hacky way to get rid of singularity on poles
            areas[0, :] = areas[-1, 0]
    return areas
