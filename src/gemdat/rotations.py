from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from gemdat.trajectory import Trajectory


@dataclass
class Orientations:
    """Container for orientational data. It computes trajectories of normalized
    unit vectors defined as the distance between a central and satellite atoms,
    meant to track orientation of moleules or clusters.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory
    center_type: str
        Type of the central atoms
    satellite_type: str
        Type of the satellite atoms
    nr_central_atoms: int
        Number of central atoms (? ask Theo ?)
    nr_ligands: optional[int]
        Number of ligands
    """
    traj: Trajectory
    center_type: str
    satellite_type: str
    nr_central_atoms: int
    nr_ligands: int | None = None

    @property
    def _traj_cent(self) -> Trajectory:
        """Return trajectory of center atoms."""
        return self.traj.filter(self.center_type)

    @property
    def _traj_sat(self) -> Trajectory:
        """Return trajectory of satellite atoms."""
        return self.traj.filter(self.satellite_type)

    def _pbc_dist(self, frac1: np.ndarray, frac2: np.ndarray) -> np.floating:
        """Computes the distance using periodic boundary conditions.

        Parameters
        ----------
        frac1: np.ndarray
            Fractional coordinates of atom1
        frac2: np.ndarray
            Fractional coordinates of atom2

        Returns
        -------
        dist: np.floating
            Distance between atom1 and atom2 considering the pbc
        """
        lattice = self.traj.lattice
        frac = np.subtract(frac2, frac1)
        frac = np.mod(frac + 0.5, 1) - 0.5
        cart = np.dot(frac, lattice)
        dist = np.linalg.norm(cart)
        return dist

    def _fractional_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return fractional coordinates of central atoms and sattelite
        atoms."""
        return self._traj_cent.positions, self._traj_sat.positions

    def _starting_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return starting coordinates of all central atoms and all satellite
        atoms."""
        return self._traj_cent.positions[1, :, :], self._traj_sat.positions[
            1, :, :]

    @property
    def _distances(self) -> np.ndarray:
        """Calculate distances between every central atom and all satellite
        atoms."""
        central_start_coord, satellite_start_coord = self._starting_coordinates(
        )
        distance = np.array([[
            self._pbc_dist(central, satellite)
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
            Matrix that shows which center-satellite pair is closer than the matchin criteria
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

    def _compute_unit_vectors_traj(self, normalize: bool) -> None:
        """Computes trajectories of normalized unit vectors defined as the
        distance between a central and satellite atoms, meant to track
        orientation of moleules or clusters.

        Parameters
        ----------
        normalize: bool
            If true, normalize the trajectories

        Returns
        -------
        unit_vec_traj: np.ndarray
            Trajectories of the unit vectors
        """
        lattice = self.traj.lattice
        direction = self.fractional_directions(self._distances)
        unit_vec_traj = np.matmul(direction, lattice)

        if normalize:
            unit_vec_traj = unit_vec_traj / np.linalg.norm(
                unit_vec_traj, axis=-1, keepdims=True)

        self._unit_vectors_traj = unit_vec_traj
        self._unit_vectors_traj_norm = normalize

    def get_unit_vectors_traj(self, normalize: bool = False) -> np.ndarray:
        """Returns trajectories of normalized unit vectors defined as the
        distance between a central and satellite atoms, meant to track
        orientation of moleules or clusters.

        Parameters
        ----------
        normalize: bool
            If true, normalize the trajectories

        Returns
        -------
        unit_vec_traj: np.ndarray
            Trajectories of the unit vectors
        """
        # Recompute also if normalized differently as expected
        if not hasattr(self, '_unit_vectors_traj'
                       ) or self._unit_vectors_traj_norm != normalize:
            self._compute_unit_vectors_traj(normalize)
        return self._unit_vectors_traj

    def _compute_conventional_form(self, normalize: bool = False) -> None:
        """Converts the trajectory of unit vectors from fractional to
        conventional coordinates.

        Parameters
        ----------
        normalize: bool
            If true, normalize the trajectories
        """
        unit_vec_traj = self.get_unit_vectors_traj(normalize=normalize)
        # Matrix to transform primitive unit cell coordinates to conventional unit cell coordinates
        prim_to_conv_matrix = np.array(
            [[1 / np.sqrt(2), -1 / np.sqrt(6), 1 / np.sqrt(3)],
             [1 / np.sqrt(2), 1 / np.sqrt(6), -1 / np.sqrt(3)],
             [0, 2 / np.sqrt(6), 1 / np.sqrt(3)]])

        self._conventional_form = np.matmul(unit_vec_traj,
                                            prim_to_conv_matrix.T)

    def get_conventional_form(self, normalize: bool = False) -> np.ndarray:
        """Returns the trajectory of unit vectors in conventional coordinates.

        Parameters
        ----------
        normalize: bool
            If true, normalize the trajectories

        Returns
        -------
        conventional_traj: np.ndarray
            Trajectory of the unit vectors in conventional coordinates (? @Theo)
        """
        if not hasattr(self, '_conventional_form'):
            self._compute_conventional_form(normalize)
        return self._conventional_form

    def apply_symmetry(self, direction: np.ndarray,
                       sym_matrix: np.ndarray) -> np.ndaray:
        """Apply symmetry elements of Oh space group with vectorized
        operations.

        Parameters
        ----------
        direction: np.ndarray
            Trajectory of the unit vectors in conventional coordinates
        sym_matrix: np.ndarray
            Matrix of symmetry operations

        Returns
        -------
        direction_sym: np.ndarray
            Trajectory of the unit vectors after applying symmetry operations
        """

        n_ts = direction.shape[0]
        n_bonds = direction.shape[1]
        n_symops = sym_matrix.shape[2]

        direction_sym = np.zeros((n_ts, n_bonds * n_symops, 3))
        for m in range(n_ts):
            for l in range(n_bonds):
                for k in range(n_symops):
                    direction_sym[m, l * n_symops + k, :] = np.matmul(
                        sym_matrix[:, :, k], direction[m, l, :])

        return direction_sym

    def _cart2sph(self, x: float, y: float,
                  z: float) -> tuple[float, float, float]:
        """Transform cartesian coordinates to spherical coordinates.

        Parameters
        ----------
        x : float
            x coordinate
        y : float
            y coordinate
        z : float
            z coordinate

        Returns
        -------
        az : float
            azimuthal angle
        el : float
            elevation angle
        r : float
            radius
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        el = np.arcsin(z / r)
        az = np.arctan2(y, x)
        return az, el, r

    def cartesian_to_spherical(self, direct_cart: np.ndarray,
                               degrees: bool) -> np.ndarray:
        """Trajectory from cartesian coordinates to spherical coordinates.

        Parameters
        ----------
        direct_cart : np.ndarray
            Trajectory of the unit vectors in conventional coordinates
        degrees : bool
            If true, return angles in degrees

        Returns
        -------
        direction_spherical : np.ndarray
            Trajectory of the unit vectors in spherical coordinates
        """
        x = direct_cart[:, :, 0]
        y = direct_cart[:, :, 1]
        z = direct_cart[:, :, 2]

        az, el, r = self._cart2sph(x, y, z)

        if degrees:
            az = np.degrees(az)
            el = np.degrees(el)

        direction_spherical = np.stack((az, el, r), axis=-1)

        return direction_spherical


@dataclass
class Oh_point_group:
    """Container for the symmetry operations of the Oh point group."""

    @property
    def sym_ops_Oh_firstcolumn(self):
        """Returns the symmetry operations for the first column."""
        return np.array([
            np.eye(3),
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        ])

    @property
    def sym_ops_Oh_firstrow(self):
        """Returns the symmetry operation for the first row."""
        return np.array([
            np.eye(3),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        ])

    @property
    def sym_ops_Oh(self):
        """Multiply row by column the sym_ops to get 8x6 matrix of
        transformation matrices."""
        sym_ops_Oh = np.zeros((8, 6, 3, 3))
        sym_ops_Oh_firstcolumn = self.sym_ops_Oh_firstcolumn
        sym_ops_Oh_firstrow = self.sym_ops_Oh_firstrow
        for i in range(len(sym_ops_Oh_firstcolumn)):
            for j in range(len(sym_ops_Oh_firstrow)):
                sym_ops_Oh[i, j, :, :] = np.matmul(
                    sym_ops_Oh_firstcolumn[i, :, :],
                    sym_ops_Oh_firstrow[j, :, :])

        # Reshape to make it easier to iterate on (only on 3rd dimension)
        return sym_ops_Oh.reshape(3, 3, -1)


def calculate_spherical_areas(shape: tuple, radius: float = 1) -> np.ndarray:
    """Calculate the areas of a section of a sphere, defined in spherical
    coordinates.

    Parameters
    ----------
    shape : tuple
        Shape of the grid
    radius : float
        Radius of the sphere

    Returns
    -------
    areas : np.ndarray
        Areas of the section
    """
    azimuthal_angles = np.linspace(0, 360, shape[1])
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


def sph_prob(direction_spherical_deg, shape=(360, 180)):

    az = direction_spherical_deg[:, :, 0].flatten()
    el = direction_spherical_deg[:, :, 1].flatten()

    # Compute the 2D histogram - for reasons, x-y inversed
    hist, xedges, yedges = np.histogram2d(el, az, shape)

    def calculate_spherical_areas(shape, radius=1):
        azimuthal_angles = np.linspace(0, 360, shape[0])
        elevation_angles = np.linspace(0, 180, shape[1])

        areas = np.zeros(shape, dtype=float)

        for i in range(shape[0]):
            for j in range(shape[1]):
                azimuthal_increment = np.deg2rad(1)
                elevation_increment = np.deg2rad(1)

                areas[i, j] = (radius**2) * azimuthal_increment * np.sin(
                    np.deg2rad(elevation_angles[j])) * elevation_increment
                #hacky way to get rid of singularity on poles
                areas[:, 0] = areas[:, -1]
        return areas

    areas = calculate_spherical_areas(shape)

    hist = np.divide(hist, areas)

    #replace values at the poles where normalization breaks - hacky
    hist[:, 0] = hist[:, 1]
    hist[:, -1] = hist[:, -2]
    return hist


def rectilinear_plot(grid):
    """Plot a rectilinear projection of a spherical function."""
    values = grid.T
    phi = np.linspace(0, 360, np.ma.size(values, 0))
    theta = np.linspace(0, 180, np.ma.size(values, 1))

    theta, phi = np.meshgrid(theta, phi)

    fig, ax = plt.subplots(subplot_kw=dict(projection='rectilinear'))
    cs = ax.contourf(phi, theta, values, cmap='viridis')
    ax.set_yticks(np.arange(0, 190, 45))
    ax.set_xticks(np.arange(0, 370, 45))

    ax.set_xlabel(r'azimuthal angle φ $[\degree$]')
    ax.set_ylabel(r'elevation θ $[\degree$]')

    ax.grid(visible=True)
    cbar = fig.colorbar(cs, label='areal probability', format='')

    # Rotate the colorbar label by 180 degrees
    cbar.ax.yaxis.set_label_coords(2.5,
                                   0.5)  # Adjust the position of the label
    cbar.set_label('areal probability', rotation=270, labelpad=15)
    return
