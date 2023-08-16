from __future__ import annotations

import typing
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from math import ceil

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.units import FloatWithUnit
from scipy.constants import Boltzmann, angstrom, elementary_charge

from .simulation_metrics import SimulationMetrics
from .transitions import Transitions
from .utils import is_lattice_similar

if typing.TYPE_CHECKING:

    from gemdat.trajectory import Trajectory

NOSITE = -1


class SitesData:

    def __init__(self,
                 *,
                 structure: Structure,
                 trajectory: Trajectory,
                 floating_specie: str,
                 n_parts: int = 10):
        """Contain sites and jumps data.

        Parameters
        ----------
        structure : Structure
            Input structure with known jump sites
        trajectory : Trajectory
            Input trajectory
        floating_specie : str
            Name of the floating or diffusing specie
        n_parts : int, optional
            Number of parts to divide transitions into for statistics
        """
        self.n_parts = n_parts

        self.floating_specie = floating_specie
        self.structure = structure
        self.trajectory = trajectory
        self.diff_trajectory = trajectory.filter(floating_specie)
        self.metrics = SimulationMetrics(self.diff_trajectory)

        self.warn_if_lattice_not_similar()

        self.transitions = Transitions.from_trajectory(
            structure=structure,
            trajectory=trajectory,
            floating_specie=floating_specie)

        self.set_split(n_parts)

    def set_split(self, n_parts: int):
        """Set number to split transitions into for statistics.

        Sets `.parts`.
        """
        self.parts = self.transitions.split(n_parts,
                                            n_steps=len(self.trajectory))

    @property
    def site_coords(self):
        """Return fractional coordinates for known sites."""
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
        """Return total number of jumps."""
        return len(self.transitions.events)

    @property
    def n_solo_jumps(self):
        """Return number of solo jumps."""
        return self.collective()[2]

    @property
    def solo_fraction(self):
        """Fraction of solo jumps."""
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

        These correspond to the axes in `.collective_matrix`.
        """
        return ['->'.join(key) for key in self.jumps()]

    @property
    def transitions_parts(self):
        return np.stack([part.matrix() for part in self.parts])

    @property
    def occupancy_parts(self):
        return [part.occupancy() for part in self.parts]

    @property
    def site_occupancy_parts(self):
        labels = self.structure.labels
        n_steps = len(self.trajectory)

        parts = self.parts

        return [
            _calculate_site_occupancy(occupancy=part.occupancy(),
                                      labels=labels,
                                      n_steps=n_steps / self.n_parts)
            for part in parts
        ]

    @property
    def atom_locations_parts(self):
        multiplier = self.n_sites / self.n_floating
        return [{
            k: v * multiplier
            for k, v in part.items()
        } for part in self.site_occupancy_parts]

    @property
    def jumps_parts(self):
        parts = self.parts

        labels = self.structure.labels
        jumps_parts = []

        for part in parts:
            jumps = Counter([(labels[i], labels[j])
                             for i, j in part.events[:, 1:3]])
            jumps_parts.append(jumps)

        return jumps_parts

    @lru_cache
    def rates(self) -> dict[tuple[str, str], tuple[float, float]]:
        """Calculate jump rates (total jumps / second).

        Returns
        -------
        rates : dict[tuple[str, str], tuple[float, float]]
            Dictionary with jump rates and standard deviations between site pairs
        """
        rates: dict[tuple[str, str], tuple[float, float]] = {}

        n_parts = self.n_parts

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

        Returns
        -------
        e_act : dict[tuple[str, str], tuple[float, float]]
            Dictionary with jump activation energies and standard deviations between site pairs.
        """
        attempt_freq, _ = self.metrics.attempt_frequency()

        e_act = {}

        n_parts = self.n_parts

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

    def site_occupancy(self):
        """Calculate percentage occupancy per unique site.

        Returns
        -------
        site_occopancy : dict[str, float]
            Percentage occupancy per unique site
        """
        labels = self.structure.labels
        n_steps = len(self.trajectory)
        return _calculate_site_occupancy(
            occupancy=self.transitions.occupancy(),
            labels=labels,
            n_steps=n_steps)

    def atom_locations(self):
        """Calculate fraction of time atoms spent at a type of site.

        Returns
        -------
        dict[str, float]
            Return dict with the fraction of time atoms spent at a site
        """
        multiplier = self.n_sites / self.n_floating
        return {k: v * multiplier for k, v in self.site_occupancy().items()}

    def jumps(self, ):
        """Calculate number of jumpst between sites.

        Returns
        -------
        jumps : dict[tuple[str, str], int]
            Dictionary with number of jumpst per sites combination
        """
        labels = self.structure.labels
        jumps = Counter([(labels[i], labels[j])
                         for i, j in self.transitions.events[:, 1:3]])

        return jumps

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

        jump_diff = np.sum(pdist**2 * self.transitions.matrix())
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
        events = self.transitions.events
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
