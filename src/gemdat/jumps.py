from __future__ import annotations

from collections import Counter
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core.units import FloatWithUnit
from scipy.constants import Boltzmann, angstrom, elementary_charge

from .caching import weak_lru_cache
from .collective import Collective
from .simulation_metrics import SimulationMetrics
from .sites import SitesData
from .transitions import Transitions, _calculate_transitions_matrix


def _generic_transitions_to_jumps(transitions):
    events = transitions.events.copy()
    events['start time'] = events['time'].copy()
    events['stop time'] = events['time'] + 1
    del events['time']

    jumps = []
    curevent = None

    # remove the 'void' between jumps
    for _, event in events.iterrows():
        if event['start site'] == -1:
            if curevent is not None:
                if curevent['atom index'] == event['atom index']:
                    event['start site'] = curevent['start site']
                    event['start time'] = curevent['start time']
                    jumps.append(event)
            curevent = None
        elif event['destination site'] == -1:
            curevent = event
        else:
            jumps.append(event)
    jumps = pd.DataFrame(data=jumps)

    # remove jumps where start == destination
    jumps = jumps[jumps['start site'] !=
                  jumps['destination site']].reset_index()

    # remove old index
    del jumps['index']
    return jumps


class Jumps:

    def __init__(self,
                 transitions: Transitions,
                 sites: SitesData,
                 conversion_method=_generic_transitions_to_jumps):
        self.transitions = transitions
        self.sites = sites
        self.conversion_method = conversion_method
        self.jumps = conversion_method(transitions)

    def as_dataframe(self):
        return self.jumps

    @property
    def n_jumps(self) -> int:
        """Return total number of jumps."""
        return len(self.jumps)

    @property
    def n_solo_jumps(self) -> int:
        """Return number of solo jumps."""
        return self.collective().n_solo_jumps

    @property
    def n_floating(self) -> int:
        """Return number of floating species."""
        return len(self.transitions.trajectory.species)

    @property
    def solo_fraction(self) -> float:
        """Fraction of solo jumps."""
        return self.n_solo_jumps / self.n_jumps

    @property
    def jump_names(self) -> list[str]:
        """Return list of jump names."""
        return ['->'.join(key) for key in self.sites.site_pairs]

    @weak_lru_cache()
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
        lattice = self.transitions.trajectory.get_lattice()
        structure = self.transitions.structure
        total_time = self.transitions.trajectory.total_time

        pdist = lattice.get_all_distances(structure.frac_coords,
                                          structure.frac_coords)

        jump_diff = np.sum(pdist**2 * self.matrix())
        jump_diff *= angstrom**2 / (2 * dimensions * self.n_floating *
                                    total_time)

        jump_diff = FloatWithUnit(jump_diff, 'm^2 s^-1')

        return jump_diff

    @weak_lru_cache()
    def matrix(self) -> np.ndarray:
        """Convert list of transition events to dense matrix.

        Returns
        -------
        transitions_matrix : np.ndarray
            Square matrix with number of each transitions
        """
        return _calculate_transitions_matrix(self.jumps,
                                             n_sites=self.transitions.n_sites)

    @weak_lru_cache()
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

        trajectory = self.transitions.trajectory
        structure = self.transitions.structure

        time_step = trajectory.time_step
        attempt_freq, _ = SimulationMetrics(trajectory).attempt_frequency()

        max_steps = ceil(1.0 / (attempt_freq * time_step))

        return Collective(
            jumps=self,
            structure=structure,
            lattice=trajectory.get_lattice(),
            max_steps=max_steps,
            max_dist=max_dist,
        )

    @weak_lru_cache()
    def activation_energies(
            self) -> dict[tuple[str, str], tuple[float, float]]:
        """Calculate activation energies for jumps (UNITS?).

        Returns
        -------
        e_act : dict[tuple[str, str], tuple[float, float]]
            Dictionary with jump activation energies and standard deviations between site pairs.
        """
        trajectory = self.transitions.trajectory
        attempt_freq, _ = SimulationMetrics(trajectory).attempt_frequency()

        e_act = {}

        temperature = trajectory.metadata['temperature']

        atom_locations_parts = self.sites.atom_locations_parts()
        jumps_cnt_parts = self.jumps_cnt_parts(self.sites.n_parts)

        for site_pair in self.sites.site_pairs:
            site_start, site_stop = site_pair

            n_jumps = np.array([part[site_pair] for part in jumps_cnt_parts])

            part_time = trajectory.total_time / self.sites.n_parts

            atom_percentage = np.array(
                [part[site_start] for part in atom_locations_parts])

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

    def jumps_cnt(self) -> Counter:
        """Calculate number of jumps between sites.

        Returns
        -------
        jumps : dict[tuple[str, str], int]
            Dictionary with number of jumps per sites combination
        """
        labels = self.transitions.structure.labels
        jumps = Counter([(labels[i], labels[j]) for _, (
            i, j) in self.jumps[['start site', 'destination site']].iterrows()
                         ])
        return jumps

    def split(self, n_parts) -> list[Any]:
        """Split the jumps into parts.

        Returns
        -------
        jumps : list[Jumps]
        """

        parts = self.transitions.split(n_parts,
                                       n_steps=len(
                                           self.transitions.trajectory))

        return [
            Jumps(part, self.sites, self.conversion_method) for part in parts
        ]

    @weak_lru_cache()
    def jumps_cnt_parts(self, n_parts) -> list[Counter]:
        """Return [jump counters][gemdat.sites.SitesData.jumps] per part."""

        return [part.jumps_cnt() for part in self.split(n_parts)]

    @weak_lru_cache()
    def rates(self) -> dict[tuple[str, str], tuple[float, float]]:
        """Calculate jump rates (total jumps / second).

        Returns
        -------
        rates : dict[tuple[str, str], tuple[float, float]]
            Dictionary with jump rates and standard deviations between site pairs
        """
        rates: dict[tuple[str, str], tuple[float, float]] = {}

        parts = self.jumps_cnt_parts(self.sites.n_parts)

        for site_pair in self.sites.site_pairs:
            n_jumps = [part[site_pair] for part in parts]

            part_time = self.transitions.trajectory.total_time / self.sites.n_parts
            denom = self.n_floating * part_time

            jump_freq_mean = np.mean(n_jumps) / denom
            jump_freq_std = np.std(n_jumps, ddof=1) / denom

            rates[site_pair] = float(jump_freq_mean), float(jump_freq_std)

        return rates
