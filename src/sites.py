from __future__ import annotations

import typing
import warnings
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from gemdat.trajectory import Trajectory
from gemdat.vibration import Vibration
from MDAnalysis.lib.pkdtree import PeriodicKDTree
from pymatgen.core import Lattice, Structure
from pymatgen.core.units import FloatWithUnit
from scipy.constants import Boltzmann, angstrom, elementary_charge

from .utils import bfill, ffill, is_lattice_similar

if typing.TYPE_CHECKING:
    pass

NOSITE = -1


class SitesData:

    def __init__(self, structure: Structure, trajectory: Trajectory,
                 vibration: Vibration):
        self.trajectory = trajectory
        self.structure = structure
        self.vibration = vibration
        lattice = self.trajectory.get_lattice()
        self.warn_if_lattice_not_similar(lattice)

    def precompute(self):
        pass

    @property
    def site_coords(self):
        return self.structure.frac_coords

    @property
    def succes(self):
        """Alias for `self.transitions_parts`."""
        return self.transitions_parts

    @property
    def n_sites(self):
        """Return number of sites."""
        return len(self.structure)

    @property
    def jump_names(self) -> list[str]:
        """Return list of jump names.

        These correspond to the axes in `.coll_matrix`.
        """
        return ['->'.join(key) for key in self.jumps()]

    def n_jumps(self) -> int:
        """Return number of jumps."""
        return len(self.all_transitions())

    def correlation_factor(self):
        """Correlation factor."""
        return self.trajectory.tracer_diffusivity() / self.jump_diffusivity()

    def solo_frac(self):
        return self.collective()[2] / len(self.all_transitions())

    def warn_if_lattice_not_similar(self, other_lattice: Lattice):
        this_lattice = self.structure.lattice

        if not is_lattice_similar(other_lattice, this_lattice):
            warnings.warn(f'Lattice mismatch: {this_lattice.parameters} '
                          f'vs. {other_lattice.parameters}')

    @lru_cache
    def dist_close(self):
        """Calculate tolerance wihin which atoms are considered to be close to
        a site.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory

        Returns
        -------
        dist_close : float
            Atoms within this distance (in Angstrom) are considered to be close to a site
        """
        lattice = self.trajectory.get_lattice()
        dist_close = 2 * self.vibration.vibration_amplitude()

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
        """Calculate nearest site for each atom coordinate.

        Note: This is a slow operation, because a pairwise distance matrix between all `coords` and
        all `site_coords` has to be generated. This includes lattice translations. The nearest site
        may be in the neighbouring unit cell.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory
        diff_coords:
            Input array with (diffusing) atom coordinates [time, atom, (x, y, z)]

        Returns
        -------
        atom_sites : np.ndarray
            Output array with site locations for each atom at each time step [time, atom].
            The value corresponds to the index in the `site_coords`.
            -1 indicates that atom is not at any site.
        """
        coords = self.trajectory.coords

        # Unit cell parameters
        lattice = self.trajectory.get_lattice()

        # Atoms within this distance (in Angstrom) are considered to be close to a site
        dist_close = self.dist_close()

        # fractional coordinates

        # Input array with site coordinates [site, (x, y, z)]
        site_cart_coords = np.dot(self.site_coords, lattice.matrix)
        site_coords_tree: PeriodicKDTree = PeriodicKDTree(
            box=np.array(lattice.parameters, dtype=np.float32))
        site_coords_tree.set_coords(site_cart_coords, cutoff=dist_close)

        atom_sites = []

        for atom_index, atom_coords in enumerate(coords.swapaxes(0, 1)):

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
        """Calculate atom transition states per time step by back/forward filling
        `self.atom_sites`.

        Returns
        -------
        atom_sites_from : np.ndarray
            Output array with atom transition states. contains
            the index of the previous site for every atom.
            If `atom_sites_from` and `atom_sites_to` are equal, the atom
            is currently sitting at that site.
        """
        return ffill(self.atom_sites(), fill_val=NOSITE, axis=0)

    @lru_cache
    def atom_sites_to(self) -> np.ndarray:
        """Calculate atom transition states per time step by back/forward filling
        `self.atom_sites`.

        Returns
        -------
        atom_sites_to : np.ndarray
            Output array with atom transition states. contains
            the index of the next site for every atom.
            If `atom_sites_from` and `atom_sites_to` are equal, the atom
            is currently sitting at that site.
        """
        return bfill(self.atom_sites(), fill_val=NOSITE, axis=0)

    @lru_cache
    def parts(self, n_parts: int) -> list[SitesData]:
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
        parts = []
        for trajectory in self.trajectory.parts(n_parts):
            vibration = Vibration(trajectory, fs=self.vibration.fs)
            sites = SitesData(trajectory=trajectory,
                              vibration=vibration,
                              structure=self.structure)
            parts.append(sites)

        return parts

    def transitions_parts(self, n_parts: int):
        return np.stack([part.transitions() for part in self.parts(n_parts)])

    def occupancy_parts(self, n_parts: int):
        return [part.occupancy() for part in self.parts(n_parts)]

    def site_occupancy_parts(self, n_parts: int):
        return [part.site_occupancy() for part in self.parts(n_parts)]

    def atom_locations_parts(self, n_parts: int):
        return [part.atom_locations() for part in self.parts(n_parts)]

    def jumps_parts(self, n_parts: int):
        return [part.jumps() for part in self.parts(n_parts)]

    def atom_locations(self) -> dict[str, float]:
        """Calculate fraction of time atoms spent at a type of site.

        Parameters
        ----------
        n_diffusing : int
            Number of diffusing atoms

        Returns
        -------
        dict[str, float]
            Return dict with the fraction of time atoms spent at a site
        """
        multiplier = self.n_sites / len(self.trajectory.species)
        return {k: v * multiplier for k, v in self.site_occupancy().items()}

    def jumps(self) -> dict[tuple[str, str], int]:
        """Calculate number of jumpst between sites.

        Returns
        -------
        jumps : dict[tuple[str, str], int]
            Dictionary with number of jumpst per sites combination
        """
        labels = self.structure.labels

        jumps = Counter([(labels[i], labels[j])
                         for i, j in self.all_transitions()[:, 1:3]])

        return jumps

    def rates(self, n_parts) -> dict[tuple[str, str], tuple[float, float]]:
        """Calculate jump rates (total jumps / second).

        Parameters
        ----------
        total_time : int
            Total time for the simulation
        n_parts : int
            total number of parts

        Returns
        -------
        rates : dict[tuple[str, str], tuple[float, float]]
            Dictionary with jump rates and standard deviations between site pairs
        """
        rates: dict[tuple[str, str], tuple[float, float]] = {}

        for site_pair in self.jumps():
            n_jumps = [part[site_pair] for part in self.jumps_parts(n_parts)]

            part_time = self.trajectory.total_time / n_parts
            denom = len(self.trajectory.species) * part_time

            jump_freq_mean = np.mean(n_jumps) / denom
            jump_freq_std = np.std(n_jumps, ddof=1) / denom

            rates[site_pair] = float(jump_freq_mean), float(jump_freq_std)

        return rates

    def activation_energies(
            self, n_parts: int) -> dict[tuple[str, str], tuple[float, float]]:
        """Calculate activation energies for jumps (UNITS?).

        Parameters
        ----------

        Returns
        -------
        e_act : dict[tuple[str, str], tuple[float, float]]
            Dictionary with jump activation energies and standard deviations between site pairs.
        n_parts : int
            total number of parts
        """
        e_act = {}

        for i, site_pair in enumerate(self.jumps()):
            site_start, site_stop = site_pair

            n_jumps = np.array(
                [part[site_pair] for part in self.jumps_parts(n_parts)])

            part_time = self.trajectory.total_time / n_parts

            atom_percentage = self.atom_locations_parts(n_parts)[i][site_start]

            denom = atom_percentage * len(self.trajectory.species) * part_time

            eff_rate = n_jumps / denom

            # For A-A jumps divide by two for a fair comparison of A-A jumps vs. A-B and B-A
            if site_start == site_stop:
                eff_rate /= 2

            e_act_arr = -np.log(
                eff_rate / self.vibration.attempt_frequency()[0]) * (
                    Boltzmann *
                    self.trajectory.temperature) / elementary_charge

            e_act[site_start,
                  site_stop] = np.mean(e_act_arr), np.std(e_act_arr, ddof=1)

        return e_act

    @lru_cache
    def jump_diffusivity(self, diffusion_dimensions: int = 3) -> float:
        """Calculate jump diffusivity.

        Parameters
        ----------
        diffusion_dimensions : int
            Number of diffusion dimensions

        Returns
        -------
        jump_diff : float
            Jump diffusivity in m^2/s
        """
        structure = self.structure

        pdist = self.trajectory.get_lattice().get_all_distances(
            structure.frac_coords, structure.frac_coords)

        jump_diff = np.sum(pdist**2 * self.transitions())
        jump_diff *= angstrom**2 / (2 * diffusion_dimensions * len(
            self.trajectory.species) * self.trajectory.total_time)

        jump_diff = FloatWithUnit(jump_diff, 'm^2 s^-1')

        return jump_diff

    @lru_cache
    def collective(self,
                   dist_collective: float = 4.5) -> tuple[list, list, int]:
        """Calculate collective jumps.

        Parameters
        ----------
        lattice : Lattice
            Lattice for distance calculations
        attempt_freq : float
            Attempt frequency in ...
        time_step : float
            Time step in seconds
        dist_collective : float, optional
            Maximum distance for collective motions in Angstrom

        Returns
        -------
        tuple[list, list, int]
            collective, list of indices to collective jump events
            coll_jumps, list with start/stop site indices involved in collective jumps
            n_solo_jumps, number of solo jumps
        """
        from math import ceil
        coll_steps = ceil(1.0 / (self.vibration.attempt_frequency()[0] *
                                 self.trajectory.time_step))

        time_col = 4
        events = self.all_transitions()
        lattice = self.trajectory.get_lattice()
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
    def collective_matrix(self, dist_collective: float = 4.5) -> np.ndarray:
        """Calculate collective jumps matrix.

        Returns
        -------
        coll_matrix : np.ndarray
            Matrix where all types of jumps combinations are counted
        dist_collective : float, optional
            Maximum distance for collective motions in Angstrom
        """
        labels = self.structure.labels

        jump_names = list(self.jumps())

        coll_matrix = np.zeros((len(jump_names), len(jump_names)), dtype=int)
        _, collective_jumps, _ = self.collective(dist_collective)

        for ((start_i, stop_i), (start_j, stop_j)) in collective_jumps:
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
        coll_sorted = np.sort(np.array(self.collective()[0]).flatten())
        difference = np.diff(coll_sorted, prepend=0)
        multi_coll = coll_sorted[difference == 0]

        return multi_coll

    @lru_cache
    def all_transitions(self) -> np.ndarray:
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

        for atom_index, atom_site in enumerate(self.atom_sites().T):

            # Indices when atom jumps to new or back to same site
            i, = np.nonzero((atom_site != np.roll(atom_site, shift=1))
                            & (atom_site >= 0))

            # Log transition events
            i_event = np.nonzero(
                atom_site[i] != np.roll(atom_site[i], shift=-1))
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

    @lru_cache
    def transitions(self) -> np.ndarray:
        """Convert list of transition events to dense transitions matrix.

        Parameters
        ----------
        all_transitions : np.ndarray
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

        transitions = np.zeros((self.n_sites, self.n_sites), dtype=int)
        idx, counts = np.unique(self.all_transitions()[:,
                                                       [start_col, stop_col]],
                                return_counts=True,
                                axis=0)
        start_idx, stop_idx = idx.T
        transitions[start_idx, stop_idx] = counts
        return transitions

    @lru_cache
    def occupancy(self) -> dict[int, int]:
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
        unq, counts = np.unique(self.atom_sites(), return_counts=True)
        occupancy = dict(zip(unq, counts))
        occupancy.pop(NOSITE, None)
        return occupancy

    @lru_cache
    def site_occupancy(self) -> dict[str, float]:
        """Calculate percentage occupancy per unique site.

        Parameters
        ----------
        occupancy : dict[int, int]
            Occupancy dict
        labels : list[str]
            Site labels

        Returns
        -------
        dict[str, float]
            Percentage occupancy per unique site
        """
        counts = defaultdict(list)

        labels = self.structure.labels

        assert all(v >= 0 for v in self.occupancy())

        for k, v in self.occupancy().items():
            label = labels[k]
            counts[label].append(v)

        div = len(labels) * len(self.trajectory)
        site_occupancies = {k: sum(v) / div for k, v in counts.items()}
        # site_occupancies['total'] = sum(chain(*counts.values())) / (len(occupancy) * n_steps)

        return site_occupancies
