"""This module contains classes for computing jumps and transitions between
sites."""

from __future__ import annotations

import typing
from collections import defaultdict
from itertools import pairwise
from typing import Optional

import numpy as np
import pandas as pd
from MDAnalysis.lib.pkdtree import PeriodicKDTree
from pymatgen.core import Structure

from .caching import weak_lru_cache
from .simulation_metrics import SimulationMetrics
from .utils import bfill, ffill

if typing.TYPE_CHECKING:

    from gemdat.trajectory import Trajectory

NOSITE = -1


class Transitions:
    """Container class for jumps and transitions between sites.

    Attributes
    ----------
    events : np.ndarray
        5-column numpy array holding all transition events
    n_sites : int
        Total number of sites
    states : np.ndarray
        For each time step, for each atom, track the index of the site it is at.
        Assingn NOSITE if the atom is in transition
    """

    def __init__(
        self,
        *,
        trajectory: Trajectory,
        sites: Structure,
        dist_close: float,
        events: pd.DataFrame,
        states: np.ndarray,
        inner_states: np.ndarray,
    ):
        """Store event data for jumps and transitions between sites.

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory of species of interest (e.g. diffusing)
            for which transitions are generated
        sites : Structure
            Structure with known sites used for calculation of events
        dist_close: float
            Custom diameter of all sites
        events : np.ndarray
            Input events
        states : np.ndarray
            Input states
        inner_states : np.ndarray
            Input states for inner sites
        """
        self.sites = sites
        self.trajectory = trajectory
        self.dist_close = dist_close
        self.states = states
        self.inner_states = inner_states
        self.events = events

    @property
    def n_floating(self) -> int:
        """Return number of floating species."""
        return len(self.trajectory.species)

    @property
    def n_states(self) -> int:
        """Return number of states."""
        return len(self.states)

    @property
    def n_events(self) -> int:
        """Return number of events."""
        return len(self.events)

    @property
    def n_sites(self) -> int:
        """Return number of sites."""
        return len(self.sites)

    @classmethod
    def from_trajectory(
        cls,
        *,
        trajectory: Trajectory,
        sites: Structure,
        floating_specie: str,
        site_radius: Optional[float] = None,
        site_inner_fraction: float = 1.,
    ) -> Transitions:
        """Compute transitions for floating specie from trajectory and
        structure with known sites.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory
        sites : pymatgen.core.structure.Structure
            Input sites with known sites
        floating_specie : str
            Name of the floating specie to calculate transitions for
        site_radius: Optional[float]
            A custom site size to use for determining if an atom is at a site
        """
        diff_trajectory = trajectory.filter(floating_specie)
        vibration_amplitude = SimulationMetrics(
            diff_trajectory).vibration_amplitude()

        if site_radius is None:
            dist_close = _dist_close(trajectory=trajectory,
                                     sites=sites,
                                     vibration_amplitude=vibration_amplitude)
        else:
            dist_close = site_radius

        states = _calculate_atom_states(sites=sites,
                                        trajectory=diff_trajectory,
                                        dist_close=dist_close)
        inner_states = _calculate_atom_states(
            sites=sites,
            trajectory=diff_trajectory,
            dist_close=dist_close,
            site_inner_fraction=site_inner_fraction)

        events = _calculate_transition_events(atom_sites=states,
                                              atom_inner_sites=inner_states)

        obj = cls(events=events,
                  states=states,
                  inner_states=inner_states,
                  sites=sites,
                  dist_close=dist_close,
                  trajectory=diff_trajectory)

        return obj

    @weak_lru_cache()
    def matrix(self) -> np.ndarray:
        """Convert list of transition events to dense matrix.

        Returns
        -------
        transitions_matrix : np.ndarray
            Square matrix with number of each transitions
        """
        return _calculate_transitions_matrix(self.events, n_sites=self.n_sites)

    @weak_lru_cache()
    def states_next(self) -> np.ndarray:
        """Calculate atom transition states per time step by backward filling
        `self.states`.

        Returns
        -------
        np.ndarray
            Output array with atom transition states. `states_next` contains
            the index of the next site for every atom.
        """
        return bfill(self.states, fill_val=NOSITE, axis=0)

    @weak_lru_cache()
    def states_prev(self) -> np.ndarray:
        """Calculate atom transition states per time step by forward filling
        `self.states`.

        Returns
        -------
        np.ndarray
            Output array with atom transition states. `states_prev` contains
            the index of the previous site for every atom.
        """
        return ffill(self.states, fill_val=NOSITE, axis=0)

    def occupancy(self) -> Structure:
        """Calculate occupancy per site.

        Returns
        -------
        structure : Structure
            Structure with occupancies set on the sites.
        """
        sites = self.sites
        states = self.states

        unq, counts = np.unique(states, return_counts=True)
        counts = counts / len(states)
        occupancies = dict(zip(unq, counts))

        species = [{
            site.specie.name: occupancies.get(i, 0)
        } for i, site in enumerate(sites)]

        return Structure(
            lattice=sites.lattice,
            species=species,
            coords=sites.frac_coords,
            site_properties=sites.site_properties,
            labels=sites.labels,
        )

    def atom_locations(self):
        """Calculate fraction of time atoms spent at a type of site.

        Returns
        -------
        dict[str, float]
            Return dict with the fraction of time atoms spent at a site
        """
        multiplier = len(self.sites) / self.n_floating

        compositions_by_label = defaultdict(list)

        for site in self.occupancy():
            compositions_by_label[site.label].append(site.species.num_atoms)

        ret = {}

        for k, v in compositions_by_label.items():
            ret[k] = (sum(v) / len(v)) * multiplier

        return ret

    def split(self, n_parts: int = 10) -> list[Transitions]:
        """Split data into equal parts in time for statistics.

        Parameters
        ----------
        n_parts : int
            Number of parts to split the data into

        Returns
        -------
        parts : list[Transitions]
            List with `Transitions` object for each part
        """
        split_states = np.array_split(self.states, n_parts)
        split_inner_states = np.array_split(self.inner_states, n_parts)
        split_events = _split_transitions_events(self.events, self.n_states,
                                                 n_parts)

        split_trajectory = self.trajectory.split(n_parts)

        parts = []

        for i in range(n_parts):
            parts.append(
                self.__class__(
                    sites=self.sites,
                    trajectory=split_trajectory[i],
                    dist_close=self.dist_close,
                    states=split_states[i],
                    inner_states=split_inner_states[i],
                    events=split_events[i],
                ))

        return parts


def _calculate_transition_events(*, atom_sites: np.ndarray,
                                 atom_inner_sites: np.ndarray) -> pd.DataFrame:
    """Find transitions between sites.

    Parameters
    ----------
    atom_sites : np.ndarray
        Input array with atom sites
    atom_inner_sites : np.ndarray
        Input array with inner atom sites

    Returns
    -------
    events : pd.DataFrame
        Output array with transition events.
        Contains 5 columns: atom index, site start, site stop, time start, time stop
    """
    events = []

    for atom_index, site in enumerate(zip(atom_sites.T, atom_inner_sites.T)):
        atom_site, atom_inner_site = site

        # Indices when atom jumps in or out of site
        i, = np.nonzero((atom_site != np.roll(atom_site, shift=-1)))

        # Indices when atom jumps in or out of inner site
        i2, = np.nonzero((atom_inner_site != np.roll(atom_inner_site,
                                                     shift=-1)))

        # Drop last event if it is on the last timestep (side effect of np.roll)
        if i[-1] == len(atom_site) - 1:
            i = i[:-1]
        if i2[-1] == len(atom_inner_site) - 1:
            i2 = i2[:-1]

        time = np.unique(np.concatenate((i, i2)))

        # Select the timestep just before the transition out of the site
        transitions = np.vstack([
            np.ones_like(time) * atom_index,
            atom_site[time],
            atom_site[time + 1],
            atom_inner_site[time],
            atom_inner_site[time + 1],
            time,
        ]).T

        events.append(transitions)

    events = np.vstack(events)
    events = pd.DataFrame(data=events,
                          columns=[
                              'atom index', 'start site', 'destination site',
                              'start inner site', 'destination inner site',
                              'time'
                          ])
    return events


def _dist_close(trajectory: Trajectory, sites: Structure,
                vibration_amplitude: float) -> float:
    """Calculate tolerance wihin which atoms are considered to be close to a
    site.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    sites : pymatgen.core.structure.Structure
        Input sites

    Returns
    -------
    dist_close : float
        Atoms within this distance (in Angstrom) are considered to be close to a site
    """
    lattice = trajectory.get_lattice()
    dist_close = 2 * vibration_amplitude

    site_coords = sites.frac_coords

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
                sites[i]
                site_j = sites[j]
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
    sites: Structure,
    trajectory: Trajectory,
    dist_close: float,
    site_inner_fraction: float = 1.,
) -> np.ndarray:
    """Calculate nearest site for each atom coordinate in the trajectory.

    Note: This is a slow operation, because a pairwise distance matrix between all `coords` and
    all `site_coords` has to be generated. This includes lattice translations. The nearest site
    may be in the neighbouring unit cell.

    Parameters
    ----------
    sites : pymatgen.core.structure.Structure
        Input sites with pre-defined sites
    trajectory : Trajectory
        Input trajectory for floating atoms
    dist_close : float
        Atoms within this distance (in Angstrom) are considered to be close to a site
    site_inner_fraction: float
        Atoms that are closer than (dist_close*site_inner_fraction) to a site, are considered
        to be in the inner site

    Returns
    -------
    _calculate_atom_states : np.ndarray
        Output array with site locations for each atom at each time step [time, atom].
        The value corresponds to the index in the `site_coords`.
        -1 indicates that atom is not at any site.
    """
    # Unit cell parameters
    lattice = trajectory.get_lattice()

    site_coords = sites.frac_coords

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
        site_index = site_coords_tree.search_tree(
            atom_cart_coords, dist_close * site_inner_fraction)

        # construct mapping
        atom_site = np.full((atom_coords.shape[0], 1), NOSITE)
        for index, site in site_index:
            atom_site[index] = site
        atom_sites.append(atom_site)

    return np.hstack(atom_sites)


def _calculate_transitions_matrix(events: pd.DataFrame,
                                  n_sites: int) -> np.ndarray:
    """Convert list of transition events to dense transitions matrix.

    Parameters
    ----------
    events : pd.DataFrame
        Input array with transition events
    n_sites : int
        Number of jump sites for diffusing element. This defines the shape of the output matrix.

    Returns
    -------
    np.ndarray
        Square matrix with number of each transitions
    """
    transitions = np.zeros((n_sites, n_sites), dtype=int)
    idx, counts = np.unique(events[['start site', 'destination site']],
                            return_counts=True,
                            axis=0)
    start_idx, stop_idx = idx.T
    transitions[start_idx, stop_idx] = counts
    return transitions


def _split_transitions_events(events: pd.DataFrame,
                              n_states: int,
                              n_parts=10,
                              split_key='time',
                              dependent_keys='time') -> list[np.ndarray]:
    """Split list of transition events into equal parts in time.

    Parameters
    ----------
    events : np.ndarray
        Input array with transition events
    n_states : int
        Number of states
    n_parts : int, optional
        Number of parts to split into
    split_key : str, optional
        Key on which to digitize dataframe
    depenent_keys: str | List[str]
        keys to normalize after split

    Returns
    -------
    transitions_parts : np.ndarray
        Sorted list of transition events split into equal parts.
        The first dimension corresponds to `n_parts`.
    """
    if len(events) < n_parts:
        raise ValueError(
            f'Not enough transitions per part to split into {n_parts}')

    bins = np.linspace(0, n_states + 1, n_parts + 1, dtype=int)

    parts = [
        events[(events[split_key] >= start)
               & (events[split_key] < stop)].copy()
        for start, stop in pairwise(bins)
    ]

    # Normalize parts to zero time index
    for offset, part in zip(bins[:-1], parts):
        part[dependent_keys] -= offset

    return parts
