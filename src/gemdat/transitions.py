"""This module contains classes for computing jumps and transitions between
sites."""

from __future__ import annotations

import typing
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from warnings import warn

import numpy as np
import pandas as pd
from MDAnalysis.lib.pkdtree import PeriodicKDTree
from pymatgen.core import Structure

from .caching import weak_lru_cache
from .metrics import TrajectoryMetrics
from .utils import bfill, ffill, integer_remap, remove_partial_occupancies_from_structure

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

    DISORDER_ERROR_MSG = (
        'Input `sites` has partial occupancies. In GEMDAT, '
        '`sites` is treated as the set of all possible sites '
        'for the floating species, and partial occupancies '
        'can lead to ambiguous site assignments. '
        'Remove partial occupancies manually, or set '
        '`remove_part_occup_from_structure=True` '
        'to do it automatically.'
    )

    def __init__(
        self,
        *,
        trajectory: Trajectory,
        diff_trajectory: Trajectory,
        sites: Structure,
        events: pd.DataFrame,
        states: np.ndarray,
        inner_states: np.ndarray,
        site_radius: SiteRadius,
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
        site_radius: SiteRadius
            site_radius used to calculate if an atom is at a site.
        """
        if not (sites.is_ordered):
            warn(self.DISORDER_ERROR_MSG, stacklevel=2)

        self.sites = sites
        self.trajectory = trajectory
        self.diff_trajectory = diff_trajectory
        self.states = states
        self.inner_states = inner_states
        self.events = events
        self.site_radius = site_radius

    @classmethod
    def from_trajectory(
        cls,
        *,
        trajectory: Trajectory,
        sites: Structure,
        floating_specie: str,
        site_radius: float | dict[str, float] | None = None,
        site_inner_fraction: float | dict[str, float] = 1.0,
        remove_part_occup_from_structure: bool = False,
        fraction_of_overlap: float = 0.0,
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
        site_inner_fraction: float | dict[str, float]
            A fraction of the site radius which is determined to be the `inner site`
            which is used in jump calculations
        remove_part_occup_from_structure: bool
            A flag to remove partial occupancies from structure
        fraction_of_overlap: float
            Fraction of allowed overlap between sites

        Returns
        -------
        transitions: Transitions
        """
        if not sites.is_ordered:
            if remove_part_occup_from_structure:
                sites = remove_partial_occupancies_from_structure(structure=sites.copy())
            else:
                warn(cls.DISORDER_ERROR_MSG, stacklevel=2)

        diff_trajectory = trajectory.filter(floating_specie)

        if site_radius is None:
            vibration_amplitude = TrajectoryMetrics(diff_trajectory).vibration_amplitude()

            site_radius_obj = SiteRadius.from_vibration_amplitude(
                trajectory=trajectory,
                sites=sites,
                vibration_amplitude=vibration_amplitude,
                inner_fraction=site_inner_fraction,
            )

        else:
            site_radius_obj = SiteRadius.from_given_radius(
                trajectory=trajectory,
                sites=sites,
                radius=site_radius,
                inner_fraction=site_inner_fraction,
                fraction_of_overlap=fraction_of_overlap,
            )

        states = _calculate_atom_states(
            sites=sites,
            trajectory=diff_trajectory,
            site_radius=site_radius_obj.radius,
            site_inner_fraction=site_radius_obj.outer_states_fraction(),
        )

        inner_states = _calculate_atom_states(
            sites=sites,
            trajectory=diff_trajectory,
            site_radius=site_radius_obj.radius,
            site_inner_fraction=site_radius_obj.inner_fraction,
        )

        events = _calculate_transition_events(atom_sites=states, atom_inner_sites=inner_states)

        obj = cls(
            sites=sites,
            trajectory=trajectory,
            diff_trajectory=diff_trajectory,
            events=events,
            states=states,
            inner_states=inner_states,
            site_radius=site_radius_obj,
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
                    site_radius=self.site_radius,
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


@dataclass
class SiteRadius:
    """Container class for sites radius.

    Attributes
    ----------
    radius: dict[str, float]
        Site radius in Angstrom
    inner_fraction: dict[str, float]
        Fraction of inner sphere
    pdist: np.ndarray
        Pairwise distance matrix between sites
    site_pairs: dict
        All site pairs for given site labels in site_radius
    min_dist: dict
        Minimal distance between given sites
    """

    radius: dict[str, float]
    inner_fraction: dict[str, float]
    pdist: np.ndarray
    site_pairs: dict
    min_dist: dict

    def radius_to_dict(self):
        if isinstance(self.radius, (int, float)):
            if isinstance(self.inner_fraction, dict):
                self.radius = {label: self.radius for label in self.inner_fraction.keys()}
            elif isinstance(self.inner_fraction, (int, float)):
                self.radius = {'': self.radius}
        elif isinstance(self.radius, dict):
            self.radius = self.radius
        else:
            raise TypeError(f'Invalid type for `site_radius`: {type(self.radius)}')

        if isinstance(self.inner_fraction, (int, float)):
            self.inner_fraction = {label: self.inner_fraction for label in self.radius.keys()}
        elif isinstance(self.inner_fraction, dict):
            self.inner_fraction = self.inner_fraction
        else:
            raise TypeError(
                f'Invalid type for `site_inner_fraction`: {type(self.inner_fraction)}'
            )

    @classmethod
    def from_given_radius(
        cls,
        *,
        trajectory: Trajectory,
        sites: Structure,
        radius: float | dict[str, float],
        inner_fraction: float | dict[str, float],
        fraction_of_overlap: float = 0.0,
    ) -> SiteRadius:
        """Create SiteRadius from given radius.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory
        sites: Structure
            Input structure with atom sites
        radius: float | int | dict
            Site radius (per site label) in Angstrom
        inner_fraction: float | int | dict
            Fraction of inner sphere
        fraction_of_overlap: float
            Fraction of allowed overlap between sites

        Returns
        -------
        site_radius_obj: SiteRadius object
        """
        if fraction_of_overlap > 1:
            raise ValueError('fraction_of_overlap must be <= 1.0')

        lattice = trajectory.get_lattice()
        site_coords = sites.frac_coords
        pdist = lattice.get_all_distances(site_coords, site_coords)

        site_radius_obj = cls(
            radius=radius,
            inner_fraction=inner_fraction,
            pdist=pdist,
        )

        site_radius_obj.radius_to_dict()
        site_radius_obj._site_pairs()
        site_radius_obj._min_dist(sites)

        factor = 1 + fraction_of_overlap

        if site_radius_obj.sites_are_overlapping(factor=factor):
            site_radius_obj.raise_if_overlapping(sites=sites, factor=factor)

        return site_radius_obj

    @classmethod
    def from_vibration_amplitude(
        cls,
        *,
        trajectory: Trajectory,
        sites: Structure,
        vibration_amplitude: float,
        inner_fraction: float | dict[str, float] = 1.0,
    ) -> SiteRadius:
        """Calculate tolerance wihin which atoms are considered to be close to
        a site.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory
        sites : pymatgen.core.structure.Structure
            Input sites
        vibration_amplitude: float
            Vibration amplitude used to calculate site radius

        Returns
        -------
        site_radius : SiteRadius
            SiteRadius dataclass
        """

        lattice = trajectory.get_lattice()
        site_radius = 2 * vibration_amplitude

        site_coords = sites.frac_coords

        pdist = lattice.get_all_distances(site_coords, site_coords)
        min_dist = np.min(pdist[np.triu_indices_from(pdist, k=1)])

        if min_dist < 2 * site_radius:
            # Sites are overlapping with the chosen site_radius,
            # making it smaller
            site_radius = (0.5 * min_dist) - 0.005

            # Two sites are within half an Angstrom of each other
            # This is NOT realistic, check/change the given site
            if site_radius * 2 < 0.5:
                idx = np.argwhere(np.triu(pdist, k=1) == min_dist)

                lines = []

                for i, j in idx:
                    site_i = sites[i]
                    site_j = sites[j]
                    lines.append('\nToo close:')
                    lines.append(
                        f'{site_i.specie.name}({i}) {site_i.label} {site_i.frac_coords}'
                    )
                    lines.append(' - ')
                    lines.append(
                        f'{site_j.specie.name}({j}) {site_i.label} {site_j.frac_coords}'
                    )

                msg = ''.join(lines)

                raise ValueError(
                    'Two sites are within half an Angstrom of each other. '
                    'This is not realistic, check/change the given sites. '
                    f'Expected: > {site_radius * 2:.4f}, '
                    f'got: {min_dist:.4f} for {msg}'
                )

        site_radius_obj = SiteRadius(
            radius=site_radius, pdist=pdist, inner_fraction=inner_fraction
        )

        site_radius_obj.radius_to_dict()
        site_radius_obj._site_pairs()
        site_radius_obj._min_dist(sites)

        return site_radius_obj

    def outer_states_fraction(self) -> dict[str, float]:
        return {label: 1.0 for label in self.radius}

    def _site_pairs(self):
        """Create site pairs with distances between them from defined
        radius."""
        from itertools import combinations_with_replacement

        pairs = list(combinations_with_replacement(self.radius.keys(), 2))
        self.site_pairs = {(i, j): self.radius[i] + self.radius[j] for (i, j) in pairs}

    def _min_dist(self, sites: Structure):
        """Minimum distance (Angstom) between sites pairs."""

        self.min_dist = {}

        site_labels = np.array(sites.labels)

        for (i, j), pair_dist in self.site_pairs.items():
            I = np.where(site_labels == i)[0]
            J = np.where(site_labels == j)[0]

            if I.size == 0 or J.size == 0:
                self.min_dist[(i, j)] = float('inf')
                continue

            sub = self.pdist[np.ix_(I, J)]

            _, i_idx, j_idx = np.intersect1d(I, J, return_indices=True)
            if i_idx.size:
                sub[i_idx, j_idx] = np.inf

            self.min_dist[(i, j)] = float(sub.min(initial=np.inf))

    def sites_are_overlapping(self, factor: float = 1.0) -> bool:
        """Return True if sites any pairwise distances are within the site
        radius."""
        for key, pair_dist in self.site_pairs.items():
            min_dist = self.min_dist[key]
            if factor * min_dist < pair_dist:
                return True
        return False

    def raise_if_overlapping(self, sites: Structure, factor: float = 1.0) -> None:
        """Raise error if sites are overlapping."""
        lines = []
        for key, pair_dist in self.site_pairs.items():
            min_dist = self.min_dist[key]
            if factor * min_dist < pair_dist:
                idx = np.argwhere(np.triu(self.pdist, k=1) == min_dist)
                for i, j in idx:
                    site_i = sites[i]
                    site_j = sites[j]
                    lines.append('\nOverlapping: ')
                    lines.append(f'{site_i.specie.name}({i}) {site_i.frac_coords}')
                    lines.append(' - ')
                    lines.append(f'{site_j.specie.name}({j}) {site_j.frac_coords}, ')
                    lines.append(f'expected < {min_dist / 2:.2f}, ')
                    lines.append(f'got {site_i.label}: {self.radius[site_i.label]} ')
                    lines.append(f'and {site_j.label}: {self.radius[site_j.label]}')
        msg = ''.join(lines)

        raise ValueError(
            'Sites are overlapping with the chosen site_radius '
            'and fraction_of_overlap, make site_radius smaller '
            f'for {msg}'
        )


def _calculate_atom_states(
    sites: Structure,
    trajectory: Trajectory,
    site_radius: dict[str, float],
    site_inner_fraction: dict[str, float],
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
    site_inner_fraction: dict[str, float]
        Atoms that are closer than (site_radius*site_inner_fraction) to a site,
        are considered to be in the inner site. Can also be a dict keyed by the
        site label to specify the radius by atom type.

    Returns
    -------
    _calculate_atom_states : np.ndarray
        Output array with site locations for each atom at each time step [time, atom].
        The value corresponds to the index in the `site_coords`.
        -1 indicates that atom is not at any site.
    """
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

    for label, radius in site_radius.items():
        if label:
            grouped = ((k, site) for k, site in enumerate(sites) if site.label == label)
            key, site_group = zip(*grouped)
            frac_coords = np.array([site.frac_coords for site in site_group])
            key = np.array(key)
        else:
            frac_coords = sites.frac_coords
            key = None

        cart_coords = lattice.get_cartesian_coords(frac_coords)
        site_index = periodic_tree.search_tree(cart_coords, radius * site_inner_fraction[label])

        if site_index.size == 0:
            warn(f'No floating species in range of {label} ({radius=})', stacklevel=2)
            continue

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
