from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.symmetry.groups import PointGroup

from gemdat.trajectory import Trajectory


def transform_normalize(orientations: Orientations):
    """Normalize the trajectory of unit vectors."""
    vectors = orientations.vectors / np.linalg.norm(
        orientations.vectors, axis=-1, keepdims=True)

    new = Orientations(
        trajectory=orientations.trajectory,
        center_type=orientations.center_type,
        satellite_type=orientations.satellite_type,
        nr_central_atoms=orientations.nr_central_atoms,
        vectors=vectors,
    )
    new.prim_to_conv_matrix = orientations.prim_to_conv_matrix
    new.sym_matrix = orientations.sym_matrix
    return new


def transform_conventional(orientations: Orientations):
    """Convert the trajectory of unit vectors from fractional to conventional
    coordinates.

    A conventional unit cell only contains one lattice point, while the
    primitive cell contains the Bravais lattice. This means that the
    conventional form is simpler to visualize and compare.

    This transformation requires the definition of the conversion matrix
    Orientations._prim_to_conv_matrix.
    """
    vectors = np.matmul(orientations.vectors,
                        orientations._prim_to_conv_matrix.T)

    new = Orientations(
        trajectory=orientations.trajectory,
        center_type=orientations.center_type,
        satellite_type=orientations.satellite_type,
        nr_central_atoms=orientations.nr_central_atoms,
        vectors=vectors,
    )
    new.sym_matrix = orientations.sym_matrix
    return new


def transform_symmetrize(orientations: Orientations):
    """Apply symmetry elements to the trajectory to improve statistics."""
    if not hasattr(orientations, 'sym_matrix'):
        raise ValueError('Symmetry operations not set')

    n_ts = orientations.vectors.shape[0]
    n_bonds = orientations.vectors.shape[1]
    n_symops = orientations.sym_matrix.shape[2]

    direction_sym = np.einsum('tbi,ijk->tbkj', orientations.vectors,
                              orientations.sym_matrix)
    vectors = direction_sym.reshape(n_ts, n_bonds * n_symops, 3)

    new = Orientations(
        trajectory=orientations.trajectory,
        center_type=orientations.center_type,
        satellite_type=orientations.satellite_type,
        nr_central_atoms=orientations.nr_central_atoms,
        vectors=vectors,
    )
    new.prim_to_conv_matrix = orientations.prim_to_conv_matrix
    return new


TRANSFORM_DISPATCH = {
    'normalize': transform_normalize,
    'conventional': transform_conventional,
    'symmetrize': transform_symmetrize,
}


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
    nr_central_atoms: int
        Number of central atoms, which corresponds to the number of cluster molecules
    vectors: np.ndarray
        Vectors representing rotation direction
    """
    trajectory: Trajectory

    center_type: str
    satellite_type: str
    nr_central_atoms: int
    vectors: np.ndarray

    def __post_init__(self):
        """Computes trajectories of unit vectors defined as the distance
        between a central and satellite atoms, meant to track orientation of
        molecules or clusters."""
        if self.vectors is None:
            lattice = self.trajectory.get_lattice()
            direction = self._fractional_directions(self._distances)
            self.vectors = lattice.get_cartesian_coords(direction)
        self._prim_to_conv_matrix = np.eye(3)
        self.sym_matrix = []

    @property
    def prim_to_conv_matrix(self):
        return self._prim_to_conv_matrix

    @prim_to_conv_matrix.setter
    def prim_to_conv_matrix(self, mat: np.ndarray):
        if mat.shape != (3, 3):
            raise ValueError('prim_to_conv_matrix must be a 3x3 matrix')
        self._prim_to_conv_matrix = mat

    @property
    def _time_step(self) -> float:
        """Return the time step of the trajectory."""
        return self.trajectory.time_step

    @property
    def _trajectory_cent(self) -> Trajectory:
        """Return trajectory of center atoms."""
        return self.trajectory.filter(self.center_type)

    @property
    def _trajectory_sat(self) -> Trajectory:
        """Return trajectory of satellite atoms."""
        return self.trajectory.filter(self.satellite_type)

    def _fractional_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return fractional coordinates of central atoms and satellite
        atoms."""
        return self._trajectory_cent.positions, self._trajectory_sat.positions

    @property
    def _distances(self) -> np.ndarray:
        """Calculate distances between every central atom and all satellite
        atoms."""
        central_start_coord = self._trajectory_cent.base_positions
        satellite_start_coord = self._trajectory_sat.base_positions
        lattice = self.trajectory.get_lattice()
        distance = np.array([[
            lattice.get_all_distances(central, satellite)
            for satellite in satellite_start_coord
        ] for central in central_start_coord])
        return distance

    def _matching_matrix(self, distance: np.ndarray,
                         frac_coord_cent: np.ndarray) -> np.ndarray:
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
            Matrix that shows which center-satellite pair is closer than the matching criteria
        """
        # The matching criteria is defined here
        match_criteria = 1.5 * np.min(distance)

        distance_match = np.where(distance < match_criteria, distance, 0)
        matching_matrix = np.zeros((len(frac_coord_cent[0, :, 0]), 4),
                                   dtype=int)
        for k in range(len(frac_coord_cent[0, :, 0])):
            matching_matrix[k, :] = np.where(distance_match[k, :] != 0)[0][:4]
        return matching_matrix

    def _central_satellite_matrix(self, distance: np.ndarray,
                                  frac_coord_cent: np.ndarray) -> np.ndarray:
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
        index_central_atoms = np.arange(self.nr_central_atoms)
        matching_matrix = self._matching_matrix(distance, frac_coord_cent)
        combinations = np.array([(i, j) for i in index_central_atoms
                                 for j in matching_matrix[i, :]])
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
        frac_coord_cent, frac_coord_sat = self._fractional_coordinates()
        combinations = self._central_satellite_matrix(distance,
                                                      frac_coord_cent)

        sat = frac_coord_sat[:, combinations[:, 1], :]
        cent = frac_coord_cent[:, combinations[:, 0], :]

        direction = sat - cent

        # Take the periodic boundary conditions into account.
        direction = np.where(direction > 0.5, direction - 1, direction)
        direction = np.where(direction < -0.5, direction + 1, direction)

        return direction

    def set_symmetry_operations(
        self,
        sym_group: str | None = None,
        explicit_sym: np.ndarray | None = None,
    ) -> None:
        """Set the symmetry operations.

        Parameters
        ----------
        sym_group: str
            Name of the symmetry group in Hermann-Mauguin notation
        explicit_sym: np.ndarray
            Matrix of symmetry operations
        """
        if sym_group is not None and explicit_sym is not None:
            raise ValueError(
                'Only one of sym_group or explicit_sym must be provided')
        elif sym_group is not None:
            g = PointGroup(sym_group)
            self.sym_matrix = np.array([
                element.rotation_matrix for element in g.symmetry_ops
            ]).transpose(1, 2, 0)
        elif explicit_sym is not None:
            if explicit_sym.ndim != 3:
                # reshape if a single transformation is provided
                explicit_sym = explicit_sym.reshape(1, 3, 3)
            self.sym_matrix = explicit_sym
        else:
            raise ValueError(
                'At least one of sym_group or explicit_sym must be provided')

    def transform(self,
                  transformations: list[str] | None = None) -> Orientations:
        """Execute all transformations in the list of transformations.

        Returns
        -------
        transformed_vectors : np.ndarray
            the unit vector trajectory after the transformations.
        """
        if not transformations:
            transformations = []

        t_operators = [TRANSFORM_DISPATCH[name] for name in transformations]

        orientations = self

        for t_operator in t_operators:
            orientations = t_operator(orientations)

        return orientations


def calculate_spherical_areas(shape: tuple[int, int],
                              radius: float = 1) -> np.ndarray:
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

            areas[j, i] = (radius**2) * azimuthal_increment * np.sin(
                np.deg2rad(elevation_angles[j])) * elevation_increment
            #hacky way to get rid of singularity on poles
            areas[0, :] = areas[-1, 0]
    return areas


def mean_squared_angular_displacement(trajectory: np.ndarray) -> np.ndarray:
    """Compute the mean squared angular displacement using FFT.

    Parameters
    ----------
    trajectory : np.ndarray
        The input signal in direct cartesian coordinates. It is expected
        to have shape (n_times, n_particles, n_coordinates)

    Returns
    -------
    msad:
        The mean squared angular displacement
    """
    n_times, n_particles, n_coordinates = trajectory.shape

    msad = np.zeros((n_particles, n_times))
    normalization = np.arange(n_times, 0, -1)

    for c in range(n_coordinates):
        signal = trajectory[:, :, c]

        # Compute the FFT of the signal
        fft_signal = np.fft.rfft(signal, n=2 * n_times - 1, axis=0)
        # Compute the power spectral density in-place
        np.square(np.abs(fft_signal), out=fft_signal)
        # Compute the inverse FFT of the power spectral density
        autocorr_c = np.fft.irfft(fft_signal, axis=0)

        # Only keep the positive times
        autocorr_c = autocorr_c[:n_times, :]

        msad += autocorr_c.T / normalization

    # Normalize the msad such that it starts from 1
    # (this makes the normalization independent on the dimensions)
    msad = msad / msad[:, 0, np.newaxis]

    return msad
