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

from .collective import Collective
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
    def site_labels(self):
        """Return site labels."""
        return self.structure.labels

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
        return self.collective().n_solo_jumps

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
        """Return list of jump names."""
        return ['->'.join(key) for key in self.site_pairs]

    @property
    def site_pairs(self) -> list[tuple[str, str]]:
        """Return list of all unique site pairs."""
        labels = self.site_labels
        return list({(label1, label2)
                     for label1 in labels
                     for label2 in labels})

    @property
    def transitions_parts(self):
        return np.stack([part.matrix() for part in self.parts])

    @property
    def occupancy_parts(self):
        return [part.occupancy() for part in self.parts]

    @property
    def site_occupancy_parts(self):
        labels = self.site_labels
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

        labels = self.site_labels
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

        for site_pair in self.site_pairs:
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

        for i, site_pair in enumerate(self.site_pairs):
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
        labels = self.site_labels
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
        labels = self.site_labels
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
    def collective(self, max_dist: float = 4.5) -> Collective:
        """Calculate collective jumps.

        Parameters
        ----------
        max_dist : float, optional
            Maximum distance for collective motions in Angstrom

        Returns
        -------
        collective : Collective
            Output class with data on collective jumps
        """
        time_step = self.trajectory.time_step
        attempt_freq, _ = self.metrics.attempt_frequency()

        max_steps = ceil(1.0 / (attempt_freq * time_step))

        return Collective(
            transitions=self.transitions,
            structure=self.structure,
            lattice=self.trajectory.get_lattice(),
            max_steps=max_steps,
            max_dist=max_dist,
        )


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
