from __future__ import annotations

import typing
from functools import lru_cache

import numpy as np
from MDAnalysis.lib.pkdtree import PeriodicKDTree
from pymatgen.core import Structure

from .simulation_metrics import SimulationMetrics
from .utils import bfill, ffill

if typing.TYPE_CHECKING:

    from gemdat.trajectory import Trajectory

NOSITE = -1


class Transitions:

    def __init__(self, events: np.ndarray, states: np.ndarray, n_sites: int):

        self.states = states
        self.events = events
        self.n_sites = n_sites

    @classmethod
    def from_trajectory(cls, *, trajectory: Trajectory, structure: Structure,
                        floating_specie: str):
        diff_trajectory = trajectory.filter(floating_specie)
        vibration_amplitude = SimulationMetrics(
            diff_trajectory).vibration_amplitude()

        dist_close = _dist_close(trajectory=trajectory,
                                 structure=structure,
                                 vibration_amplitude=vibration_amplitude)

        states = _calculate_atom_states(structure=structure,
                                        trajectory=diff_trajectory,
                                        dist_close=dist_close)
        events = _calculate_transition_events(atom_sites=states)

        obj = cls(events=events, states=states, n_sites=len(structure))

        obj._dist_close = dist_close  # type: ignore

        return obj

    @lru_cache
    def matrix(self):
        """Convert list of transition events to dense matrix.

        Returns
        -------
        transitions_matrix : np.ndarray
            Square matrix with number of each transitions
        """
        return _calculate_transitions_matrix(self.events, n_sites=self.n_sites)

    @lru_cache
    def states_next(self):
        """Calculate atom transition states per time step by backward filling
        `self.states`.

        Returns
        -------
        np.ndarray
            Output array with atom transition states. `states_next` contains
            the index of the next site for every atom.
        """
        return bfill(self.states, fill_val=NOSITE, axis=0)

    @lru_cache
    def states_prev(self):
        """Calculate atom transition states per time step by forward filling
        `self.states`.

        Returns
        -------
        np.ndarray
            Output array with atom transition states. `states_prev` contains
            the index of the previous site for every atom.
        """
        return ffill(self.states, fill_val=NOSITE, axis=0)

    def occupancy(self):
        """Calculate occupancy per site.

        Returns
        -------
        occupancy : dict[int, int]
            For each site, count for how many time steps it is occupied by an atom
        """
        return _calculate_occupancy(self.states)

    def split(self, n_parts: int = 10, *, n_steps: int):
        """Split data into equal parts in time for statistics.

        Parameters
        ----------
        n_parts : int
            Number of parts to split the data into

        Returns
        -------
        parts : list[SitesData]
            List with `Transitions` object for each part
        """
        split_states = np.split(self.states, n_parts)
        split_events = _split_transitions_events(self.events, n_steps, n_parts)

        kwargs_list = []

        for states, events in zip(split_states, split_events):
            kwargs_list.append({
                'states': states,
                'events': events,
                'n_sites': self.n_sites,
            })

        return [self.__class__(**kwargs) for kwargs in kwargs_list]


def _calculate_transition_events(*, atom_sites: np.ndarray) -> np.ndarray:
    """Find transitions between sites.

    Parameters
    ----------
    atom_sites : np.ndarray
        Input array with atom sites

    Returns
    -------
    events : np.ndarray
        Output array with transition events.
        Contains 5 columns: atom index, time start, time stop, site start, site stop
    """
    events = []

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
        events.append(transitions)

    return np.vstack(events)


def _dist_close(trajectory: Trajectory, structure: Structure,
                vibration_amplitude: float) -> float:
    """Calculate tolerance wihin which atoms are considered to be close to a
    site.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    structure : Structure
        Input structure

    Returns
    -------
    dist_close : float
        Atoms within this distance (in Angstrom) are considered to be close to a site
    """
    lattice = trajectory.get_lattice()
    dist_close = 2 * vibration_amplitude

    site_coords = structure.frac_coords

    pdist = lattice.get_all_distances(site_coords, site_coords)
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
                structure.sites[i]
                site_j = structure.sites[j]
                lines.append('\nToo close:')
                lines.append(
                    '{site_i.specie.name}({i}) {site_i.frac_coords} - ')
                lines.append(f'{site_j.specie.name}({j}) {site_j.frac_coords}')

            msg = ''.join(lines)

            raise ValueError(
                f'Crystallographic sites are too close together (expected: >{dist_close*2:.4f}, '
                f'got: {min_dist:.4f} for {msg}')

    return dist_close


def _calculate_atom_states(
    structure: Structure,
    trajectory: Trajectory,
    dist_close: float,
) -> np.ndarray:
    """Calculate nearest site for each atom coordinate in the trajectory.

    Note: This is a slow operation, because a pairwise distance matrix between all `coords` and
    all `site_coords` has to be generated. This includes lattice translations. The nearest site
    may be in the neighbouring unit cell.

    Parameters
    ----------
    structure : Structure
        Input structure with pre-defined sites
    trajectory : Trajectory
        Input trajectory for floating atoms
    dist_close : float
        Atoms within this distance (in Angstrom) are considered to be close to a site

    Returns
    -------
    _calculate_atom_states : np.ndarray
        Output array with site locations for each atom at each time step [time, atom].
        The value corresponds to the index in the `site_coords`.
        -1 indicates that atom is not at any site.
    """
    # Unit cell parameters
    lattice = trajectory.get_lattice()

    site_coords = structure.frac_coords

    # Input array with site coordinates [site, (x, y, z)]
    site_cart_coords = np.dot(site_coords, lattice.matrix)
    site_coords_tree: PeriodicKDTree = PeriodicKDTree(
        box=np.array(lattice.parameters, dtype=np.float32))
    site_coords_tree.set_coords(site_cart_coords, cutoff=dist_close)

    atom_sites = []

    for atom_index, atom_coords in enumerate(
            trajectory.positions.swapaxes(0, 1)):

        # index and distance of nearest site
        atom_cart_coords = np.dot(atom_coords, lattice.matrix)
        site_index = site_coords_tree.search_tree(atom_cart_coords, dist_close)

        # construct mapping
        atom_site = np.full((atom_coords.shape[0], 1), NOSITE)
        for index, site in site_index:
            atom_site[index] = site

        atom_sites.append(atom_site)

    return np.hstack(atom_sites)


def _calculate_transitions_matrix(events: np.ndarray,
                                  n_sites: int) -> np.ndarray:
    """Convert list of transition events to dense transitions matrix.

    Parameters
    ----------
    events : np.ndarray
        Input array with transition events
    n_sites : int
        Number of jump sites for diffusing element. This defines the shape of the output matrix.

    Returns
    -------
    np.ndarray
        Square matrix with number of each transitions
    """
    start_col = 1  # transition starts
    stop_col = 2  # transition stop

    transitions = np.zeros((n_sites, n_sites), dtype=int)
    idx, counts = np.unique(events[:, [start_col, stop_col]],
                            return_counts=True,
                            axis=0)
    start_idx, stop_idx = idx.T
    transitions[start_idx, stop_idx] = counts
    return transitions


def _split_transitions_events(events: np.ndarray,
                              n_steps: int,
                              n_parts=10) -> list[np.ndarray]:
    """Split list of transition events into equal parts in time.

    Parameters
    ----------
    events : np.ndarray
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
    parts = np.digitize(events[:, col], bins=bins)
    parts = parts[parts.argsort()]
    splits = np.unique(parts, return_index=True)[1][1:]

    sorted_transitions = events[events[:, col].argsort()]

    parts = np.split(sorted_transitions, splits)

    if len(parts) < n_parts:
        raise ValueError(
            f'Not enough transitions per part to split into {n_parts}')

    return parts


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
