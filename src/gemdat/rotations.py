from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.symmetry.groups import PointGroup

from gemdat.trajectory import Trajectory


@dataclass
class Orientations:
    """Container for orientational data. It computes trajectories of normalized
    unit vectors defined as the distance between a central and satellite atoms,
    meant to track orientation of molecules or clusters.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory
    center_type: str
        Type of the central atoms
    satellite_type: str
        Type of the satellite atoms
    nr_central_atoms: int
        Number of central atoms, which corresponds to the number of cluster molecules
    nr_ligands: optional[int]
        Number of ligands
    """
    trajectory: Trajectory

    center_type: str
    satellite_type: str
    nr_central_atoms: int
    normalize_traj: bool = False

    @property
    def _traj_cent(self) -> Trajectory:
        """Return trajectory of center atoms."""
        return self.traj.filter(self.center_type)

    @property
    def _traj_sat(self) -> Trajectory:
        """Return trajectory of satellite atoms."""
        return self.traj.filter(self.satellite_type)

    def _fractional_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return fractional coordinates of central atoms and satellite
        atoms."""
        return self._traj_cent.positions, self._traj_sat.positions

    @property
    def _distances(self) -> np.ndarray:
        """Calculate distances between every central atom and all satellite
        atoms."""
        central_start_coord = self._traj_cent.base_positions
        satellite_start_coord = self._traj_sat.base_positions
        lattice = self.traj.get_lattice()
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
        frac_coord_cent; np.ndarray
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

    def fractional_directions(self, distance: np.ndarray) -> np.ndarray:
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
        direction = frac_coord_sat[:,
                                   combinations[:,
                                                1], :] - frac_coord_cent[:,
                                                                         combinations[:,
                                                                                      0], :]
        # Take the periodic boundary conditions into account.
        direction = np.where(direction > 0.5, direction - 1, direction)
        direction = np.where(direction < -0.5, direction + 1, direction)
        return direction

    def _compute_unit_vectors_traj(self) -> None:
        """Computes trajectories of normalized unit vectors defined as the
        distance between a central and satellite atoms, meant to track
        orientation of molecules or clusters."""
        lattice = self.traj.lattice
        direction = self.fractional_directions(self._distances)
        unit_vec_traj = np.matmul(direction, lattice)

        if self.normalize_traj:
            unit_vec_traj = unit_vec_traj / np.linalg.norm(
                unit_vec_traj, axis=-1, keepdims=True)

        return unit_vec_traj

    def get_unit_vectors_traj(self) -> np.ndarray:
        """Returns trajectories of normalized unit vectors defined as the
        distance between a central and satellite atoms, meant to track
        orientation of molecules or clusters.

        Returns
        -------
        unit_vec_traj: np.ndarray
            Trajectories of the unit vectors
        """
        # Recompute also if normalized differently as expected
        if not hasattr(self, '_unit_vectors_traj'):
            self._unit_vectors_traj = self._compute_unit_vectors_traj()
        return self._unit_vectors_traj


    def _compute_conventional_coordinates(self, ) -> None:
        """Converts the trajectory of unit vectors from fractional to
        conventional coordinates.

        A conventional unit cell only contains one lattice point, while
        the primitive cell contains the Bravais lattice. This means that
        the conventional form is simpler to visualize and compare.
        """
        unit_vec_traj = self.get_unit_vectors_traj()
        # Matrix to transform primitive unit cell coordinates to conventional unit cell coordinates
        prim_to_conv_matrix = np.array(
            [[1 / np.sqrt(2), -1 / np.sqrt(6), 1 / np.sqrt(3)],
             [1 / np.sqrt(2), 1 / np.sqrt(6), -1 / np.sqrt(3)],
             [0, 2 / np.sqrt(6), 1 / np.sqrt(3)]])

        return np.matmul(unit_vec_traj,
                                                   prim_to_conv_matrix.T)

    def get_conventional_coordinates(self, ) -> np.ndarray:
        """Returns the trajectory of unit vectors in conventional coordinates.
        Conventional coordinates are the coordinates of the unit vectors in the
        conventional unit cell.

        Returns
        -------
        conventional_traj: np.ndarray
            Trajectory of the unit vectors in conventional coordinates
        """
        if not hasattr(self, '_conventional_coordinates'):
            self._conventional_coordinates = self._compute_conventional_coordinates()
        return self._conventional_coordinates


    def _compute_symmetric_traj(self, ) -> None:
        """Apply symmetry elements to the trajectory to improve statistics.

        It requires the unit vectors trajectory in conventional
        coordinates.
        """
        direction = self.get_conventional_coordinates()

        n_ts = direction.shape[0]
        n_bonds = direction.shape[1]
        n_symops = self.sym_matrix.shape[2]

        direction_sym = np.einsum('tbi,ijk->tbkj', direction, self.sym_matrix)
        direction_sym = direction_sym.reshape(n_ts, n_bonds * n_symops, 3)

        self._symmetric_traj = direction_sym

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
            self.sym_matrix = explicit_sym
        else:
            raise ValueError(
                'At least one of sym_group or explicit_sym must be provided')

    def get_symmetric_traj(self, ) -> np.ndarray:
        """Returns the symmetric trajectory.

        Returns
        -------
        direction_sym: np.ndarray
            Trajectory of the unit vectors after applying symmetry operations
        """
        if not hasattr(self, '_symmetric_traj'):
            self._compute_symmetric_traj()
        return self._symmetric_traj


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

    for i in range(shape[1]):
        for j in range(shape[0]):
            azimuthal_increment = np.deg2rad(1)
            elevation_increment = np.deg2rad(1)

            areas[j, i] = (radius**2) * azimuthal_increment * np.sin(
                np.deg2rad(elevation_angles[j])) * elevation_increment
            #hacky way to get rid of singularity on poles
            areas[0, :] = areas[-1, 0]
    return areas


def autocorrelation(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the autocorrelation of the trajectory using FFT.

    Parameters
    ----------
    traj : np.ndarray
        The input signal in direct cartesian coordinates. It is expected
        to have shape (n_times, n_particles, n_coordinates)

    Returns
    -------
    autocorr.mean : np.ndarray
        The autocorrelation of the signal mediated over the number of particles.
    autocorr.std : np.ndarray
        The standard deviation of the autocorrelation of the signal.
    """

    n_times, n_particles, n_coordinates = traj.shape

    # Sum the coordinates to get the magnitude of the signal
    signal = np.sum(traj, axis=2)

    # Compute the FFT of the magnitude
    fft_magnitude = np.fft.fft(signal, n=2 * n_times - 1, axis=0)

    # Compute the power spectral density
    psd = np.abs(fft_magnitude)**2

    # Compute the inverse FFT of the power spectral density
    autocorr = np.fft.ifft(psd, axis=0)

    # Only keep the positive time lags
    autocorr = autocorr[:n_times, :]

    # Normalize
    autocorr = autocorr.T
    autocorr /= np.arange(n_times, 0, -1)

    # and get the real part
    autocorr = autocorr.real

    return autocorr.mean(axis=0), autocorr.std(axis=0)
