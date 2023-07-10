from __future__ import annotations

import warnings
from collections import defaultdict
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core import Lattice, Structure

from .utils import bfill, ffill

if TYPE_CHECKING:
    from gemdat.data import SimulationData

NOSITE = -1


def lattice_is_similar(a: Lattice,
                       b: Lattice,
                       length_tol: float = 0.5,
                       angle_tol: float = 0.5) -> bool:
    """Return True if lattices are similar within given tolerance.

    Parameters
    ----------
    a, b : Lattice
        Input lattices
    length_tol : float, optional
        Length tolerance in Angstrom
    angle_tol : float, optional
        Angle tolerance in degrees

    Returns
    -------
    bool
        Return True if lattices are similar
    """
    for a_length, b_length in zip(a.lengths, b.lengths):
        if abs(a_length - b_length) > length_tol:
            return False

    for a_angle, b_angle in zip(a.angles, b.angles):
        if abs(a_angle - b_angle) > angle_tol:
            return False

    return True


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

    def warn_if_lattice_not_similar(self, other_lattice: Lattice):
        this_lattice = self.structure.lattice

        if not lattice_is_similar(other_lattice, this_lattice):
            warnings.warn(f'Lattice mismatch: {this_lattice.parameters} '
                          'vs. {other_lattice.parameters}')

    def calculate_all(self, data: SimulationData, extras: SimpleNamespace):
        """Calculate all parameters.

        Parameters
        ----------
        data : SimulationData
            Input simulation data
        extras : SimpleNamespace
            Extra parameters
        """
        self.warn_if_lattice_not_similar(data.structure.lattice)

        self.dist_close = self.calculate_dist_close(
            data, vibration_amplitude=extras.vibration_amplitude)
        self.atom_sites = self.calculate_atom_sites(
            data, diff_coords=extras.diff_coords)
        self.atom_sites_to, self.atom_sites_from = self.calculate_atom_sites_transitions(
        )
        self.all_transitions = self.calculate_transition_events()
        self.transitions = self.calculate_transitions_matrix(
            n_diffusing=extras.n_diffusing)
        self.transitions_parts = self.calculate_transitions_matrix_parts(
            n_steps=extras.n_steps,
            n_diffusing=extras.n_diffusing,
            n_parts=extras.n_parts)
        self.occupancy = self.calculate_occupancy()
        self.occupancy_parts = self.calculate_occupancy_parts(
            n_parts=extras.n_parts)

        self.sites_occupancy = self.calculate_site_occupancy(
            n_steps=extras.n_steps)
        self.sites_occupancy_parts = self.calculate_site_occupancy_parts(
            n_steps=extras.n_steps / extras.n_parts)

        self.atom_locations = self.sites_occupancy  # Is this correct? TODO: check
        self.atom_locations_parts = self.sites_occupancy_parts  # Is this correct? TODO: check

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

            # Site index when close, NOSITE when in transition
            atom_site = np.where(is_at_site, nearest, NOSITE)

            atom_sites.append(atom_site)

        return np.hstack(atom_sites)

    def calculate_atom_sites_transitions(
            self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate atom transition states per time step by back/forward filling
        `self.atom_sites`.

        Returns
        -------
        atom_sites_from, atom_sites_to : tuple[np.ndarray, np.ndarray]
            Output arrays with atom transition states. `atom_sites_from` contains
            the index of the previous site for every atom, and `atom_sites_to` the
            next site. If `atom_sites_from` and `atom_sites_to` are equal, the atom
            is currently sitting at that site.
        """
        atom_sites_from = ffill(self.atom_sites, fill_val=NOSITE, axis=0)
        atom_sites_to = bfill(self.atom_sites, fill_val=NOSITE, axis=0)

        return atom_sites_from, atom_sites_to

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

    def calculate_site_occupancy(self, *, n_steps: int) -> dict[str, float]:
        """Calculate percentage occupancy per unique site.

        Parameters
        ----------
        n_steps : int
            Number of steps in time series

        Returns
        -------
        site_occopancy : dict[str, float]
            Percentage occupancy per unique site
        """
        labels = self.structure.site_properties['label']
        return _calculate_site_occupancy(occupancy=self.occupancy,
                                         labels=labels,
                                         n_steps=n_steps)

    def calculate_site_occupancy_parts(self, *,
                                       n_steps: int) -> list[dict[str, float]]:
        """Calculate percentage occupancy per unique site per part.

        Parameters
        ----------
        n_steps : int
            Number of steps in time series

        Returns
        -------
        site_occopancy : list[dict[str, float]]
            Return a list of dicts, where each dict contains the percentage occupancy per unique site
        """
        labels = self.structure.site_properties['label']
        return [
            _calculate_site_occupancy(occupancy=occupancy,
                                      labels=labels,
                                      n_steps=n_steps)
            for occupancy in self.occupancy_parts
        ]

    def calculate_transition_events(self):
        """Find transitions between sites.

        Returns
        -------
        all_transitions : np.ndarray
            Output array with transition events.
            Contains 5 columns: atom index, time start, time stop, site start, site stop
        """
        return _calculate_transition_events(atom_sites=self.atom_sites)

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


def _calculate_transition_events(*, atom_sites: np.ndarray) -> np.ndarray:
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
    occupancy : dict[int, int]
        For each site, count for how many time steps it is occupied by an atom
    """
    unq, counts = np.unique(atom_sites, return_counts=True)
    occupancy = dict(zip(unq, counts))
    occupancy.pop(NOSITE, None)
    return occupancy


def _calculate_site_occupancy(*, occupancy: dict[int, int], labels: list[str],
                              n_steps: int) -> dict[str, float]:
    """Calculate percentage occupancy per unique site.

    Parameters
    ----------
    occupancy : dict[int, int]
        Occupancy dict
    labels : list[str]
        Site labels
    n_steps : int
        Number of steps in time series

    Returns
    -------
    dict[str, float]
        Percentage occupancy per unique site
    """
    counts = defaultdict(list)

    assert all(v >= 0 for v in occupancy)

    for k, v in occupancy.items():
        label = labels[k]
        counts[label].append(v)

    div = len(labels) * n_steps
    site_occupancies = {k: sum(v) / div for k, v in counts.items()}
    # site_occupancies['total'] = sum(chain(*counts.values())) / (len(occupancy) * n_steps)

    return site_occupancies
