"""This module contains classes for computing jumps and transitions between
sites."""

from __future__ import annotations

import typing
from collections import defaultdict
from itertools import pairwise
from warnings import warn

import numpy as np
import pandas as pd
from MDAnalysis.lib.pkdtree import PeriodicKDTree
from pymatgen.core import Structure

from .caching import weak_lru_cache
from .metrics import TrajectoryMetrics
from .utils import bfill, ffill, integer_remap

if typing.TYPE_CHECKING:
    from gemdat.jumps import Jumps
    from gemdat.rdf import RDFCollection
    from gemdat.trajectory import Trajectory


NOSITE = -1


class Transitions:
    """Container class for transitions between sites.

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
        diff_trajectory: Trajectory,
        sites: Structure,
        events: pd.DataFrame,
        states: np.ndarray,
        inner_states: np.ndarray,
    ):
        """Store event data for jumps and transitions between sites.

        Parameters
        ----------
        trajectory : Trajectory
            Full trajectory of all sites in the simulation
        diff_trajectory : Trajectory
            Trajectory of species of interest (e.g. diffusing)
            for which transitions are generated
        sites : Structure
            Structure with known sites used for calculation of events
        events : np.ndarray
            Input events
        states : np.ndarray
            Input states
        inner_states : np.ndarray
            Input states for inner sites
        """
        if not (sites.is_ordered):
            warn(
                'Input `sites` are disordered! '
                'Although the code may work, it was written under the assumption '
                'that an ordered structure would be passed. '
                'See https://github.com/GEMDAT-repos/GEMDAT/issues/339 for more information.',
                stacklevel=2,
            )

        self.sites = sites
        self.trajectory = trajectory
        self.diff_trajectory = diff_trajectory
        self.states = states
        self.inner_states = inner_states
        self.events = events

    @classmethod
    def from_trajectory(
        cls,
        *,
        trajectory: Trajectory,
        sites: Structure,
        floating_specie: str,
        site_radius: float | dict[str, float] | None = None,
        site_inner_fraction: float = 1.0,
    ) -> Transitions:
        """Compute transitions between given sites for floating specie.

        Parameters
        ----------
        sites : pymatgen.core.structure.Structure
            Input structure with known sites
        floating_specie : str
            Name of the floating specie to calculate transitions for
        site_radius: float | dict[str, float] | None
            A custom site radius in Ångstrom to determine
            if an atom is at a site. A dict keyed by the site label can
            be used to have a site per atom type, e.g.
            `site_radius = {'Li1': 1.0, 'Li2': 1.2}.
        site_inner_fraction:
            A fraction of the site radius which is determined to be the `inner site`
            which is used in jump calculations

        Returns
        -------
        transitions: Transitions
        """
        diff_trajectory = trajectory.filter(floating_specie)

        if site_radius is None:
            vibration_amplitude = TrajectoryMetrics(diff_trajectory).vibration_amplitude()

            site_radius = _compute_site_radius(
                trajectory=trajectory,
                sites=sites,
                vibration_amplitude=vibration_amplitude,
            )

        if isinstance(site_radius, float):
            site_radius = {'': site_radius}

        states = _calculate_atom_states(
            sites=sites,
            trajectory=diff_trajectory,
            site_radius=site_radius,
        )

        inner_states = _calculate_atom_states(
            sites=sites,
            trajectory=diff_trajectory,
            site_radius=site_radius,
            site_inner_fraction=site_inner_fraction,
        )

        events = _calculate_transition_events(atom_sites=states, atom_inner_sites=inner_states)

        obj = cls(
            sites=sites,
            trajectory=trajectory,
            diff_trajectory=diff_trajectory,
            events=events,
            states=states,
            inner_states=inner_states,
        )

        return obj

    def jumps(self, **kwargs) -> Jumps:
        """Analyze transitions and classify them as jumps.

        Parameters
        ----------
        **kwargs : dict
            These parameters are passed to the [gemdat.Jumps][] initializer.

        Returns
        -------
        jumps : Jumps
        """
        from gemdat.jumps import Jumps

        return Jumps(self, **kwargs)

    def radial_distribution(self, **kwargs) -> dict[str, RDFCollection]:
        """Calculate and sum RDFs for the floating species in the given sites
        data.

        Parameters
        ----------
        **kwargs : dict
            These parameters are passed to the [gemdat.radial_distribution][] function.


        Returns
        -------
        rdfs : dict[str, RDFCollection]
            Dictionary with rdf arrays per symbol
        """
        from gemdat.rdf import radial_distribution

        return radial_distribution(transitions=self, **kwargs)

    @property
    def n_floating(self) -> int:
        """Return number of floating species."""
        return len(self.diff_trajectory.species)

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
        sites : Structure
            Structure with occupancies set on the sites.
        """
        sites = self.sites
        states = self.states

        unq, counts = np.unique(states, return_counts=True)
        counts = counts / len(states)
        occupancies = dict(zip(unq, counts))

        species = [
            {site.species.elements[0].name: occupancies.get(i, 0)}
            for i, site in enumerate(sites)
        ]

        return Structure(
            lattice=sites.lattice,
            species=species,
            coords=sites.frac_coords,
            site_properties=sites.site_properties,
            labels=sites.labels,
        )

    def occupancy_by_site_type(self) -> dict[str, float]:
        """Calculate average occupancy per a type of site.

        Returns
        -------
        occupancy : dict[str, float]
            Return dict with average occupancy per site type
        """
        compositions_by_label = defaultdict(list)

        for site in self.occupancy():
            compositions_by_label[site.label].append(site.species.num_atoms)

        return {k: sum(v) / len(v) for k, v in compositions_by_label.items()}

    def atom_locations(self) -> dict[str, float]:
        """Calculate fraction of time atoms spent at a type of site.

        Returns
        -------
        dict[str, float]
            Return dict with the fraction of time atoms spent at a site
        """
        n = self.n_floating
        compositions_by_label = defaultdict(list)

        for site in self.occupancy():
            compositions_by_label[site.label].append(site.species.num_atoms)

        return {k: sum(v) / n for k, v in compositions_by_label.items()}

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
        split_events = _split_transitions_events(self.events, self.n_states, n_parts)

        split_trajectory = self.trajectory.split(n_parts)
        split_diff_trajectory = self.diff_trajectory.split(n_parts)

        parts = []

        for i in range(n_parts):
            parts.append(
                self.__class__(
                    sites=self.sites,
                    trajectory=split_trajectory[i],
                    diff_trajectory=split_diff_trajectory[i],
                    states=split_states[i],
                    inner_states=split_inner_states[i],
                    events=split_events[i],
                )
            )

        return parts


def _calculate_transition_events(
    *,
    atom_sites: np.ndarray,
    atom_inner_sites: np.ndarray,
) -> pd.DataFrame:
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
        (i,) = np.nonzero((atom_site != np.roll(atom_site, shift=-1)))

        # continue if atom does not jump
        if len(i) < 1:
            continue

        # Indices when atom jumps in or out of inner site
        (i2,) = np.nonzero((atom_inner_site != np.roll(atom_inner_site, shift=-1)))

        # Drop last event if it is on the last timestep (side effect of np.roll)
        if i[-1] == len(atom_site) - 1:
            i = i[:-1]
        if i2[-1] == len(atom_inner_site) - 1:
            i2 = i2[:-1]

        time = np.unique(np.concatenate((i, i2)))

        # Select the timestep just before the transition out of the site
        transitions = np.vstack(
            [
                np.ones_like(time) * atom_index,
                atom_site[time],
                atom_site[time + 1],
                atom_inner_site[time],
                atom_inner_site[time + 1],
                time,
            ]
        ).T

        events.append(transitions)

    events = np.vstack(events)
    events = pd.DataFrame(
        data=events,
        columns=[
            'atom index',
            'start site',
            'destination site',
            'start inner site',
            'destination inner site',
            'time',
        ],
    )
    return events


def _compute_site_radius(
    trajectory: Trajectory, sites: Structure, vibration_amplitude: float
) -> float:
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
    site_radius : float
        Atoms within this distance (in Angstrom) are considered to be close to a site
    """
    lattice = trajectory.get_lattice()
    site_radius = 2 * vibration_amplitude

    site_coords = sites.frac_coords

    pdist = lattice.get_all_distances(site_coords, site_coords)
    min_dist = np.min(pdist[np.triu_indices_from(pdist, k=1)])

    if min_dist < 2 * site_radius:
        # Crystallographic sites are overlapping with the chosen site_radius,
        # making it smaller
        site_radius = (0.5 * min_dist) - 0.005

        # Two crystallographic sites are within half an Angstrom of each other
        # This is NOT realistic, check/change the given crystallographic site
        if site_radius * 2 < 0.5:
            idx = np.argwhere(pdist == min_dist)

            lines = []

            for i, j in idx:
                site_i = sites[i]
                site_j = sites[j]
                lines.append('\nToo close:')
                lines.append(f'{site_i.specie.name}({i}) {site_i.frac_coords}')
                lines.append(' - ')
                lines.append(f'{site_j.specie.name}({j}) {site_j.frac_coords}')

            msg = ''.join(lines)

            raise ValueError(
                'Crystallographic sites are too close together '
                f'(expected: >{site_radius * 2:.4f}, '
                f'got: {min_dist:.4f} for {msg}'
            )

    return site_radius


def _calculate_atom_states(
    sites: Structure,
    trajectory: Trajectory,
    site_radius: dict[str, float],
    site_inner_fraction: float = 1.0,
) -> np.ndarray:
    """Calculate nearest site for each atom coordinate in the trajectory.

    Note: This is a slow operation, because a pairwise distance matrix between
    all `coords` and all `site_coords` has to be generated. This includes
    lattice translations. The nearest site may be in the neighbouring unit cell.

    Parameters
    ----------
    sites : pymatgen.core.structure.Structure
        Input sites with pre-defined sites
    trajectory : Trajectory
        Input trajectory for floating atoms
    site_radius : dict[str, float]
        Atoms within this distance (in Angstrom) are considered to be close to a site.
        Can also be a dict keyed by the site label to specify the radius by atom type.
    site_inner_fraction: float
        Atoms that are closer than (site_radius*site_inner_fraction) to a site,
        are considered to be in the inner site

    Returns
    -------
    _calculate_atom_states : np.ndarray
        Output array with site locations for each atom at each time step [time, atom].
        The value corresponds to the index in the `site_coords`.
        -1 indicates that atom is not at any site.
    """

    def _site_radius_iterator():
        for label, radius in site_radius.items():
            if label:
                grouped = ((k, site) for k, site in enumerate(sites) if site.label == label)
                key, site_group = zip(*grouped)
                frac_coords = np.array([site.frac_coords for site in site_group])
                yield frac_coords, np.array(key), radius
            else:
                yield sites.frac_coords, None, radius

    lattice = trajectory.get_lattice()

    cutoff = max(list(site_radius.values()))

    traj_frac_coords = trajectory.positions.reshape(-1, 3)
    traj_cart_coords = lattice.get_cartesian_coords(traj_frac_coords)

    periodic_tree: PeriodicKDTree = PeriodicKDTree(
        box=np.array(lattice.parameters, dtype=np.float32)
    )
    periodic_tree.set_coords(traj_cart_coords, cutoff=cutoff)

    shape = trajectory.positions.shape[0:2]

    atom_sites = np.full((traj_cart_coords.shape[0]), NOSITE)

    for coords, key, radius in _site_radius_iterator():
        cart_coords = lattice.get_cartesian_coords(coords)
        site_index = periodic_tree.search_tree(cart_coords, radius * site_inner_fraction)

        siteno, index = site_index.T

        if key is not None:
            siteno = integer_remap(a=siteno, key=key, palette=np.unique(siteno))

        atom_sites[index] = siteno

    return atom_sites.reshape(shape)


def _calculate_transitions_matrix(events: pd.DataFrame, n_sites: int) -> np.ndarray:
    """Convert list of transition events to dense transitions matrix.

    Parameters
    ----------
    events : pd.DataFrame
        Input array with transition events
    n_sites : int
        Number of jump sites for diffusing element.
        This defines the shape of the output matrix.

    Returns
    -------
    np.ndarray
        Square matrix with number of each transitions
    """
    transitions = np.zeros((n_sites, n_sites), dtype=int)
    idx, counts = np.unique(
        events[['start site', 'destination site']], return_counts=True, axis=0
    )
    start_idx, stop_idx = idx.T
    transitions[start_idx, stop_idx] = counts
    return transitions


def _split_transitions_events(
    events: pd.DataFrame,
    n_states: int,
    n_parts=10,
    split_key='time',
    dependent_keys='time',
) -> list[np.ndarray]:
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
        raise ValueError(f'Not enough transitions per part to split into {n_parts}')

    bins = np.linspace(0, n_states + 1, n_parts + 1, dtype=int)

    parts = [
        events[(events[split_key] >= start) & (events[split_key] < stop)].copy()
        for start, stop in pairwise(bins)
    ]

    # Normalize parts to zero time index
    for offset, part in zip(bins[:-1], parts):
        part[dependent_keys] -= offset

    return parts
