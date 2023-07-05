from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core import Structure

if TYPE_CHECKING:
    from gemdat.data import SimulationData


class SitesData:

    def __init__(self, structure: Structure):
        self.structure = structure

    @property
    def site_coords(self):
        return self.structure.frac_coords

    @property
    def succes(self):
        """Alias for `self.transitions_parts`."""
        return self.transitions_parts

    def calculate_all(self, data: SimulationData, extras: SimpleNamespace):
        """Calculate all parameters.

        Parameters
        ----------
        data : SimulationData
            Input simulation data
        extras : SimpleNamespace
            Extra parameters
        """
        self.dist_close = self.calculate_dist_close(
            data, vibration_amplitude=extras.vibration_amplitude)
        self.atom_sites = self.calculate_atom_sites(
            data, diff_coords=extras.diff_coords)
        self.all_transitions = self.calculate_transitions()
        self.transitions = self.calculate_transitions_matrix(
            n_diffusing=extras.n_diffusing)
        self.transitions_parts = self.calculate_transitions_matrix_parts(
            n_steps=extras.n_steps,
            n_diffusing=extras.n_diffusing,
            n_parts=extras.n_parts)
        self.occupancy = self.calculate_occupancy()
        self.occupancy_parts = self.calculate_occupancy_parts(
            n_parts=extras.n_parts)

    def calculate_dist_close(self, data: SimulationData,
                             vibration_amplitude: float):
        """Calculate tolerance wihin which atoms are considered to be close to
        a site.

        Parameters
        ----------
        data : SimulationData
            Simulation data
        vibration_amplitude : float
            Vibration amplitude

        Returns
        -------
        dist_close : float
            Atoms within this distance (in Angstrom) are considered to be close to a site
        """
        dist_close = 2 * vibration_amplitude

        pdist = data.lattice.get_all_distances(self.site_coords,
                                               self.site_coords)
        min_dist = np.min(pdist[np.triu_indices_from(pdist, k=1)])

        if min_dist < 2 * dist_close:
            # Crystallographic sites are overlapping with the chosen dist_close, making it smaller
            dist_close = (0.5 * min_dist) - 0.005

            # Two crystallographic sites are within half an Angstrom of each other
            # This is NOT realistic, check/change the given crystallographic site
            if dist_close * 2 < 0.5:
                idx = np.argwhere(pdist == min_dist)

                lines = []

                for i, j in idx:
                    self.structure.sites[i]
                    site_j = self.structure.sites[j]
                    lines.append('\nToo close:')
                    lines.append(
                        '{site_i.specie.name}({i}) {site_i.frac_coords} - ')
                    lines.append(
                        f'{site_j.specie.name}({j}) {site_j.frac_coords}')

                msg = ''.join(lines)

                raise ValueError(
                    f'Crystallographic sites are too close together (expected: >{dist_close*2:.4f}, '
                    f'got: {min_dist:.4f} for {msg}')

        return dist_close

    def calculate_atom_sites(self, data: SimulationData,
                             diff_coords: np.ndarray):
        """Calculate nearest site for each atom coordinate.

        Note: This is a slow operation, because a pairwise distance matrix between all `coords` and
        all `site_coords` has to be generated. This includes lattice translations. The nearest site
        may be in the neighbouring unit cell.

        Parameters
        ----------
        data : SimulationData
            Simulation data
        diff_coords:
            Input array with (diffusing) atom coordinates [time, atom, (x, y, z)]

        Returns
        -------
        atom_sites : np.ndarray
            Output array with site locations for each atom at each time step [time, atom].
            The value corresponds to the index in the `site_coords`.
            -1 indicates that atom is not at any site.
        """
        coords = diff_coords

        # Unit cell parameters
        lattice = data.lattice

        # Atoms within this distance (in Angstrom) are considered to be close to a site
        dist_close = self.dist_close

        # Input array with site coordinates [site, (x, y, z)]
        site_coords = self.site_coords

        atom_sites = []

        for atom_index, atom_coords in enumerate(coords.swapaxes(0, 1)):
            pdist = lattice.get_all_distances(atom_coords, site_coords)

            # index of nearest site
            nearest = pdist.argmin(axis=1, keepdims=True)

            # True if atom is close enough to a site
            is_at_site = np.take_along_axis(pdist, nearest,
                                            axis=1) < dist_close

            # Site index when close, -1 when in transition
            atom_site = np.where(is_at_site, nearest, -1)

            atom_sites.append(atom_site)

        return np.hstack(atom_sites)

    def calculate_occupancy(self):
        """Calculate occupancy per site.

        Returns
        -------
        occupancy: dict[int, int]
            For each site, count for how many time steps it is occupied by an atom
        """
        return _calculate_occupancy(self.atom_sites)

    def calculate_occupancy_parts(self, n_parts: int) -> list[dict[int, int]]:
        """Calculate occupancy per site, divided in parts.

        Parameters
        ----------
        n_parts : int, optional
            Number of parts to split into

        Returns
        -------
        occupancy_parts: list[dict[int, int]]
            Returns a list of dicts, where each dict corresponds to a part of the data.
            Eech dict counts the number of time steps a site is occupied by an atom
        """
        split_atom_sites = np.split(self.atom_sites, n_parts)
        return [_calculate_occupancy(part) for part in split_atom_sites]

    def calculate_transitions(self):
        """Find transitions between sites.

        Returns
        -------
        all_transitions : np.ndarray
            Output array with transition events.
            Contains 5 columns: atom index, time start, time stop, site start, site stop
        """
        return _calculate_transitions(atom_sites=self.atom_sites)

    def calculate_transitions_matrix(self, n_diffusing: int):
        """Convert list of transition events to dense transitions matrix.

        Parameters
        ----------
        n_diffusing : int
            Number of diffusing elements. This defines the shape of the output matrix.

        Returns
        -------
        transitions_matrix : np.ndarray
            Square matrix with number of each transitions
        """
        return _calculate_transitions_matrix(self.all_transitions,
                                             n_diffusing=n_diffusing)

    def calculate_transitions_matrix_parts(self, *, n_steps: int,
                                           n_diffusing: int,
                                           n_parts: int) -> np.ndarray:
        """Divide list of transition events in equal parts and convert to dense
        transition matrices.

        Note: equivalent to `sites.succes` in the matlab code.

        Parameters
        ----------
        n_steps : int
            Number of steps
        n_diffusing : int
            Number of diffusing elements. This defines the shape of the output matrix.
        n_parts : int
            Number of parts to divide the transitions events list into

        Returns
        -------
        transitions_matrix_parts : np.ndarray
            Stacked square matrices with number of each transitions.
            The number of stacked matrices is equal to the number of parts specified.
        """
        split_transitions = _split_transitions_in_parts(
            self.all_transitions, n_steps, n_parts)
        return np.stack([
            _calculate_transitions_matrix(part, n_diffusing=n_diffusing)
            for part in split_transitions
        ])


def _calculate_transitions(*, atom_sites: np.ndarray) -> np.ndarray:
    """Find transitions between sites.

    Parameters
    ----------
    atom_sites : np.ndarray
        Input array with atom sites

    Returns
    -------
    all_transitions : np.ndarray
        Output array with transition events.
        Contains 5 columns: atom index, time start, time stop, site start, site stop
    """
    all_transitions = []

    for atom_index, atom_site in enumerate(atom_sites.T):

        # Indices when atom jumps to new or back to same site
        i, = np.nonzero((atom_site != np.roll(atom_site, shift=1))
                        & (atom_site >= 0))

        # Log transition events
        i_event = np.nonzero(atom_site[i] != np.roll(atom_site[i], shift=-1))
        time_start = i[i_event]
        time_stop = np.roll(i, shift=-1)[i_event]
        transitions = np.vstack([
            np.ones_like(time_start) * atom_index,
            atom_site[time_start],
            atom_site[time_stop],
            time_start,
            time_stop,
        ]).T

        # Drop last event (side effect of np.roll)
        transitions = transitions[:-1]
        all_transitions.append(transitions)

    return np.vstack(all_transitions)


def _calculate_transitions_matrix(all_transitions: np.ndarray,
                                  n_diffusing: int) -> np.ndarray:
    """Convert list of transition events to dense transitions matrix.

    Parameters
    ----------
    all_transitions : np.ndarray
        Input array with transition events
    n_diffusing : int
        Number of diffusing elements. This defines the shape of the output matrix.

    Returns
    -------
    np.ndarray
        Square matrix with number of each transitions
    """
    start_col = 1  # transition starts
    stop_col = 2  # transition stop

    transitions = np.zeros((n_diffusing, n_diffusing), dtype=int)
    idx, counts = np.unique(all_transitions[:, [start_col, stop_col]],
                            return_counts=True,
                            axis=0)
    start_idx, stop_idx = idx.T
    transitions[start_idx, stop_idx] = counts
    return transitions


def _split_transitions_in_parts(all_transitions: np.ndarray,
                                n_steps: int,
                                n_parts=10) -> list[np.ndarray]:
    """Split list of transition events into equal parts in time.

    Parameters
    ----------
    all_transitions : np.ndarray
        Input array with transition events
    n_steps : int
        Number of time steps
    n_parts : int, optional
        Number of parts to split into

    Returns
    -------
    transitions_parts : np.ndarray
        Sorted list of transition events split into equal parts.
        The first dimension corresponds to `n_parts`.
    """
    col = 4

    bins = np.linspace(0, n_steps + 1, n_parts + 1, dtype=int)
    parts = np.digitize(all_transitions[:, col], bins=bins)
    parts = parts[parts.argsort()]
    splits = np.unique(parts, return_index=True)[1][1:]

    sorted_transitions = all_transitions[all_transitions[:, col].argsort()]

    return np.split(sorted_transitions, splits)


def _calculate_occupancy(atom_sites: np.ndarray) -> dict[int, int]:
    """Calculate occupancy per site.

    Parameters
    ----------
    atom_sites : np.ndarray
        Input array with atom sites

    Returns
    -------
    dict[int, int]
        For each site, count for how many time steps it is occupied by an atom
    """
    unq, counts = np.unique(atom_sites, return_counts=True)
    return dict(zip(unq, counts))
