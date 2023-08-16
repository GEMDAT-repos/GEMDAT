from __future__ import annotations

import typing
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from math import ceil

import numpy as np
from MDAnalysis.lib.pkdtree import PeriodicKDTree
from pymatgen.core import Structure
from pymatgen.core.units import FloatWithUnit
from scipy.constants import Boltzmann, angstrom, elementary_charge

from .simulation_metrics import SimulationMetrics
from .utils import bfill, ffill, is_lattice_similar

if typing.TYPE_CHECKING:

    from gemdat.trajectory import Trajectory

NOSITE = -1


class SitesData:

    def __init__(self, *, structure: Structure, trajectory: Trajectory,
                 floating_specie: str):

        self.floating_specie = floating_specie
        self.structure = structure
        self.trajectory = trajectory
        self.diff_trajectory = trajectory.filter(floating_specie)
        self.metrics = SimulationMetrics(self.diff_trajectory)

        self.dist_close = self._dist_close()

        self.warn_if_lattice_not_similar()

    @property
    def site_coords(self):
        return self.structure.frac_coords

    @property
    def n_sites(self):
        """Return number of sites."""
        return len(self.structure)

    @property
    def n_floating(self):
        """Return number of floating species."""
        return len(self.diff_trajectory.species)

    @property
    def n_jumps(self):
        return len(self.transition_events())

    @property
    def n_solo_jumps(self):
        return self.collective()[2]

    @property
    def solo_fraction(self):
        return self.n_solo_jumps / self.n_jumps

    def warn_if_lattice_not_similar(self):
        """Raise warning if structure and trajectory lattices do not match."""
        this_lattice = self.structure.lattice
        other_lattice = self.trajectory.get_lattice()

        if not is_lattice_similar(other_lattice, this_lattice):
            warnings.warn(f'Lattice mismatch: {this_lattice.parameters} '
                          f'vs. {other_lattice.parameters}')

    @property
    def jump_names(self) -> list[str]:
        """Return list of jump names.

        These correspond to the axes in `.coll_matrix`.
        """
        return ['->'.join(key) for key in self.jumps()]

    def _dist_close(self):
        """Calculate tolerance wihin which atoms are considered to be close to
        a site.

        Returns
        -------
        dist_close : float
            Atoms within this distance (in Angstrom) are considered to be close to a site
        """
        lattice = self.trajectory.get_lattice()
        dist_close = 2 * self.metrics.vibration_amplitude()

        pdist = lattice.get_all_distances(self.site_coords, self.site_coords)
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

    @lru_cache
    def atom_sites(self) -> np.ndarray:
        """Calculate nearest site for each atom coordinate in the trajectory.

        Note: This is a slow operation, because a pairwise distance matrix between all `coords` and
        all `site_coords` has to be generated. This includes lattice translations. The nearest site
        may be in the neighbouring unit cell.

        Returns
        -------
        atom_sites : np.ndarray
            Output array with site locations for each atom at each time step [time, atom].
            The value corresponds to the index in the `site_coords`.
            -1 indicates that atom is not at any site.
        """
        # Unit cell parameters
        lattice = self.trajectory.get_lattice()

        # Atoms within this distance (in Angstrom) are considered to be close to a site
        dist_close = self.dist_close

        # Input array with site coordinates [site, (x, y, z)]
        site_cart_coords = np.dot(self.site_coords, lattice.matrix)
        site_coords_tree: PeriodicKDTree = PeriodicKDTree(
            box=np.array(lattice.parameters, dtype=np.float32))
        site_coords_tree.set_coords(site_cart_coords, cutoff=dist_close)

        atom_sites = []

        for atom_index, atom_coords in enumerate(
                self.diff_trajectory.positions.swapaxes(0, 1)):

            # index and distance of nearest site
            atom_cart_coords = np.dot(atom_coords, lattice.matrix)
            site_index = site_coords_tree.search_tree(atom_cart_coords,
                                                      dist_close)

            # construct mapping
            atom_site = np.full((atom_coords.shape[0], 1), NOSITE)
            for index, site in site_index:
                atom_site[index] = site

            atom_sites.append(atom_site)

        return np.hstack(atom_sites)

    @lru_cache
    def atom_sites_from(self) -> np.ndarray:
        """Calculate atom transition states per time step by forward filling
        `self.atom_sites`.

        Returns
        -------
        atom_sites_from : np.ndarray
            Output arrays with atom transition states. `atom_sites_from` contains
            the index of the previous site for every atom.
        """
        return ffill(self.atom_sites(), fill_val=NOSITE, axis=0)

    @lru_cache
    def atom_sites_to(self) -> np.ndarray:
        """Calculate atom transition states per time step by backward filling
        `self.atom_sites`.

        Returns
        -------
        atom_sites_from, atom_sites_to : np.ndarray
            Output arrays with atom transition states. `atom_sites_to` contains
            the index of the next site for every atom.
        """
        return bfill(self.atom_sites(), fill_val=NOSITE, axis=0)

    @lru_cache
    def occupancy(self):
        """Calculate occupancy per site.

        Returns
        -------
        occupancy: dict[int, int]
            For each site, count for how many time steps it is occupied by an atom
        """
        return _calculate_occupancy(self.atom_sites())

    def split(self, *, n_parts: int) -> list[SitesData]:
        """Split data into equal parts in time for internal statistics.

        Parameters
        ----------
        n_parts : int
            Number of parts to split the data into

        Returns
        -------
        parts : list[SitesData]
            List with `SitesData` object for each part
        """
        n_steps = len(self.trajectory)

        split_atom_sites = np.split(self.atom_sites, n_parts)
        split_transitions = _split_transitions_in_parts(
            self.transition_events(), n_steps, n_parts)

        parts = [SitesData(self.structure) for _ in range(n_parts)]

        for i, part in enumerate(parts):
            part.dist_close = self.dist_close
            part.atom_sites = split_atom_sites[i]
            part.atom_sites_to = part.atom_sites_to()
            part.atom_sites_from = part.atom_sites_from()

            part.transition_events = split_transitions[i]

            part.transitions_matrix = part.transitions_matrix()

            part.occupancy = part.occupancy()

            part.site_occupancy = part.site_occupancy(n_steps=int(n_steps /
                                                                  n_parts))

            part.atom_locations = part.atom_locations()

            part.jumps = part.jumps()

        return parts

    @property
    def transitions_parts(self):
        return np.stack([part.transitions() for part in self.parts])

    @property
    def occupancy_parts(self):
        return [part.occupancy() for part in self.parts]

    @property
    def site_occupancy_parts(self):
        return [part.site_occupancy() for part in self.parts]

    @property
    def atom_locations_parts(self):
        return [part.atom_locations() for part in self.parts]

    @property
    def jumps_parts(self):
        return [part.jumps() for part in self.parts]

    @lru_cache
    def site_occupancy(self, ) -> dict[str, float]:
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
        n_steps = len(self.trajectory)
        labels = self.structure.labels
        return _calculate_site_occupancy(occupancy=self.occupancy(),
                                         labels=labels,
                                         n_steps=n_steps)

    @lru_cache
    def atom_locations(self) -> dict[str, float]:
        """Calculate fraction of time atoms spent at a type of site.

        Returns
        -------
        dict[str, float]
            Return dict with the fraction of time atoms spent at a site
        """
        multiplier = self.n_sites / self.n_floating
        return {k: v * multiplier for k, v in self.site_occupancy().items()}

    @lru_cache
    def transition_events(self):
        """Find transitions between sites.

        Returns
        -------
        transition_events : np.ndarray
            Output array with transition events.
            Contains 5 columns: atom index, time start, time stop, site start, site stop
        """
        return _calculate_transition_events(atom_sites=self.atom_sites())

    @lru_cache
    def transitions_matrix(self):
        """Convert list of transition events to dense transitions matrix.

        Returns
        -------
        transitions_matrix : np.ndarray
            Square matrix with number of each transitions
        """
        return _calculate_transitions_matrix(self.transition_events(),
                                             n_sites=self.n_sites)

    @lru_cache
    def jumps(self) -> dict[tuple[str, str], int]:
        """Calculate number of jumpst between sites.

        Returns
        -------
        jumps : dict[tuple[str, str], int]
            Dictionary with number of jumpst per sites combination
        """
        labels = self.structure.labels

        jumps = Counter([(labels[i], labels[j])
                         for i, j in self.transition_events()[:, 1:3]])

        return jumps

    @lru_cache
    def rates(self) -> dict[tuple[str, str], tuple[float, float]]:
        """Calculate jump rates (total jumps / second).

        Returns
        -------
        rates : dict[tuple[str, str], tuple[float, float]]
            Dictionary with jump rates and standard deviations between site pairs
        """
        rates: dict[tuple[str, str], tuple[float, float]] = {}

        n_parts = len(self.parts)

        for site_pair in self.jumps():
            n_jumps = [part[site_pair] for part in self.jumps_parts]

            part_time = self.trajectory.total_time / n_parts
            denom = self.n_floating * part_time

            jump_freq_mean = np.mean(n_jumps) / denom
            jump_freq_std = np.std(n_jumps, ddof=1) / denom

            rates[site_pair] = float(jump_freq_mean), float(jump_freq_std)

        return rates

    @lru_cache
    def activation_energies(
            self) -> dict[tuple[str, str], tuple[float, float]]:
        """Calculate activation energies for jumps (UNITS?).

        Parameters
        ----------
        attempt_freq : float
            Jump attempt frequency

        Returns
        -------
        e_act : dict[tuple[str, str], tuple[float, float]]
            Dictionary with jump activation energies and standard deviations between site pairs.
        """
        attempt_freq, _ = self.metrics.attempt_frequency()

        e_act = {}

        n_parts = len(self.parts)

        temperature = self.trajectory.metadata['temperature']

        for i, site_pair in enumerate(self.jumps()):
            site_start, site_stop = site_pair

            n_jumps = np.array([part[site_pair] for part in self.jumps_parts])

            part_time = self.trajectory.total_time / n_parts

            atom_percentage = self.atom_locations_parts[i][site_start]

            denom = atom_percentage * self.n_floating * part_time

            eff_rate = n_jumps / denom

            # For A-A jumps divide by two for a fair comparison of A-A jumps vs. A-B and B-A
            if site_start == site_stop:
                eff_rate /= 2

            e_act_arr = -np.log(eff_rate / attempt_freq) * (
                Boltzmann * temperature) / elementary_charge

            e_act[site_start,
                  site_stop] = np.mean(e_act_arr), np.std(e_act_arr, ddof=1)

        return e_act

    @lru_cache
    def jump_diffusivity(self, dimensions: int) -> float:
        """Calculate jump diffusivity.

        Parameters
        ----------
        dimensions : int
            Number of diffusion dimensions

        Returns
        -------
        jump_diff : float
            Jump diffusivity in m^2/s
        """
        lattice = self.trajectory.get_lattice()
        structure = self.structure
        total_time = self.trajectory.total_time

        pdist = lattice.get_all_distances(structure.frac_coords,
                                          structure.frac_coords)

        jump_diff = np.sum(pdist**2 * self.transitions_matrix())
        jump_diff *= angstrom**2 / (2 * dimensions * self.n_floating *
                                    total_time)

        jump_diff = FloatWithUnit(jump_diff, 'm^2 s^-1')

        return jump_diff

    @lru_cache
    def collective(self,
                   dist_collective: float = 4.5) -> tuple[list, list, int]:
        """Calculate collective jumps.

        Parameters
        ----------
        dist_collective : float, optional
            Maximum distance for collective motions in Angstrom

        Returns
        -------
        tuple[list, list, int]
            collective, list of indices to collective jump events
            coll_jumps, list with start/stop site indices involved in collective jumps
            n_solo_jumps, number of solo jumps
        """
        lattice = self.trajectory.get_lattice()
        time_step = self.trajectory.time_step
        attempt_freq, _ = self.metrics.attempt_frequency()

        coll_steps = ceil(1.0 / (attempt_freq * time_step))

        time_col = 4
        events = self.transition_events()
        event_order = events[:, time_col].argsort()

        coll_jumps = []
        collective = []

        n_solo_jumps = 0

        structure = self.structure

        for i, event_i in enumerate(event_order[:-1]):

            for j, event_j in enumerate(event_order[i + 1:], start=1):

                atom_i, start_site_i, stop_site_i, _, stop_time_i = events[
                    event_i]
                atom_j, start_site_j, stop_site_j, _, stop_time_j = events[
                    event_j]

                if stop_time_j - stop_time_i > coll_steps:
                    break

                if atom_i == atom_j:
                    n_solo_jumps += 1
                    continue

                a = structure.frac_coords[[start_site_i, stop_site_i]]
                b = structure.frac_coords[[start_site_j, stop_site_j]]

                dists = lattice.get_all_distances(a, b)

                if np.any(dists < dist_collective):
                    collective.append((event_i, event_j))
                    coll_jumps.append(((start_site_i, stop_site_i),
                                       (start_site_j, stop_site_j)))

                else:
                    n_solo_jumps += 1

        return collective, coll_jumps, n_solo_jumps

    @lru_cache
    def collective_matrix(self) -> np.ndarray:
        """Calculate collective jumps matrix.

        Returns
        -------
        coll_matrix : np.ndarray
            Matrix where all types of jumps combinations are counted
        """
        labels = self.structure.labels
        _, coll_jumps, _ = self.collective()

        jump_names = list(self.jumps())

        coll_matrix = np.zeros((len(jump_names), len(jump_names)), dtype=int)

        for ((start_i, stop_i), (start_j, stop_j)) in coll_jumps:
            name_start_i = labels[start_i]
            name_stop_i = labels[stop_i]
            name_start_j = labels[start_j]
            name_stop_j = labels[stop_j]

            i = jump_names.index((name_start_i, name_stop_i))
            j = jump_names.index((name_start_j, name_stop_j))

            coll_matrix[i, j] += 1

        return coll_matrix

    @lru_cache
    def multiple_collective(self) -> np.ndarray:
        """Find jumps that occur collectively multiple times.

        Only returns non-unique jumps

        Returns
        -------
        multi_coll : np.ndarray
            Dictionary with indices of sites between which jumps happen and their counts.
        """
        collective, *_ = self.collective()
        coll_sorted = np.sort(np.array(collective).flatten())
        difference = np.diff(coll_sorted, prepend=0)
        multi_coll = coll_sorted[difference == 0]

        return multi_coll


def _calculate_transition_events(*, atom_sites: np.ndarray) -> np.ndarray:
    """Find transitions between sites.

    Parameters
    ----------
    atom_sites : np.ndarray
        Input array with atom sites

    Returns
    -------
    transition_events : np.ndarray
        Output array with transition events.
        Contains 5 columns: atom index, time start, time stop, site start, site stop
    """
    transition_events = []

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
        transition_events.append(transitions)

    return np.vstack(transition_events)


def _calculate_transitions_matrix(transition_events: np.ndarray,
                                  n_sites: int) -> np.ndarray:
    """Convert list of transition events to dense transitions matrix.

    Parameters
    ----------
    transition_events : np.ndarray
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
    idx, counts = np.unique(transition_events[:, [start_col, stop_col]],
                            return_counts=True,
                            axis=0)
    start_idx, stop_idx = idx.T
    transitions[start_idx, stop_idx] = counts
    return transitions


def _split_transitions_in_parts(transition_events: np.ndarray,
                                n_steps: int,
                                n_parts=10) -> list[np.ndarray]:
    """Split list of transition events into equal parts in time.

    Parameters
    ----------
    transition_events : np.ndarray
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
    parts = np.digitize(transition_events[:, col], bins=bins)
    parts = parts[parts.argsort()]
    splits = np.unique(parts, return_index=True)[1][1:]

    sorted_transitions = transition_events[transition_events[:, col].argsort()]

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
