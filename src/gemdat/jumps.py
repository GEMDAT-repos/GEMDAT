from __future__ import annotations

from collections import Counter
from itertools import product
from math import ceil
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from pymatgen.core.units import FloatWithUnit
from scipy.constants import Boltzmann, angstrom, elementary_charge

from .caching import weak_lru_cache
from .collective import Collective
from .simulation_metrics import SimulationMetrics
from .transitions import Transitions, _calculate_transitions_matrix

if TYPE_CHECKING:
    from mypy_extensions import DefaultNamedArg


def _generic_transitions_to_jumps(transitions: Transitions,
                                  *,
                                  minimal_residence: int = 0) -> pd.DataFrame:
    """Generic function to convert transition events to jumps.

    Parameters
    ----------
    transitions : Transitions
        Input transitions
    minimal_residence : int
        minimal residence time of an atom on a destination site to count as a jump
    """
    events = transitions.events.copy()
    events['stop time'] = events['time'] + 1
    events = events.rename(columns={'time': 'start time'})

    jumps = []
    fromevent = None
    candidate_jump = None

    for _, event in events.iterrows():
        # If we are jumping, but we go to the next atom index, reset
        if fromevent is not None:
            if fromevent['atom index'] != event['atom index']:
                fromevent = None

        # If we have a candidate jump, we must make sure it remains on the site
        # for minimal_residence timesteps, this is that check, add it to the jumps
        # if it passes
        if candidate_jump is not None:
            if candidate_jump['atom index'] != event['atom index']:
                jumps.append(candidate_jump)
                candidate_jump = None
            elif event['start time'] - candidate_jump[
                    'stop time'] >= minimal_residence:
                jumps.append(candidate_jump)
                candidate_jump = None
                fromevent = None
            elif candidate_jump['destination site'] != event[
                    'destination site']:
                candidate_jump = None

        # Specify the start of a jump if we encounter one
        if event['start site'] != -1:
            if event['start site'] != event['destination site']:
                fromevent = event

        if fromevent is not None:
            # if we jump back, remove fromevent
            if fromevent['start site'] == event['destination site']:
                fromevent = None
                candidate_jump = None
                continue

            # Check if jump to the inner site, add it to the jumps immediately
            if event['destination inner site'] != -1:
                event['start site'] = fromevent['start site']
                event['start time'] = fromevent['start time']
                fromevent = None
                candidate_jump = None
                jumps.append(event)
                continue

            # If we enter another site, create a candidate jump
            if candidate_jump is None:
                if event['destination site'] != -1:
                    event['start site'] = fromevent['start site']
                    event['start time'] = fromevent['start time']
                    candidate_jump = event

    # Also add a last candidate jump (if there is one
    if candidate_jump is not None:
        jumps.append(candidate_jump)

    jumps = pd.DataFrame(data=jumps)

    # remove jumps where start == destination
    jumps = jumps[jumps['start site'] !=
                  jumps['destination site']].reset_index()

    # remove old index
    del jumps['index']
    del jumps['start inner site']
    del jumps['destination inner site']
    return jumps


class Jumps:

    def __init__(self,
                 transitions: Transitions,
                 *,
                 conversion_method: Callable[
                     [Transitions,
                      DefaultNamedArg(int, 'minimal_residence')],
                     pd.DataFrame] = _generic_transitions_to_jumps,
                 minimal_residence: int = 0):
        """Analyze transitions and classify them as jumps.

        Parameters
        ----------
        transitions : Transitions
            pymatgen transitions in which to calculate jumps
        conversion_method : Callable[[Transitions,int], pd.DataFrame]:
            conversion method that translates the Transitions into Jumps,
            second parameter is the `minimal_residence` parameter
        minimal_residence : int
            minimal residence, number of timesteps that an atom needs to reside
            on a destination site to count as a jump, passed through to conversion
            method
        """
        self.transitions = transitions
        self.trajectory = transitions.diff_trajectory
        self.sites = transitions.sites
        self.conversion_method = conversion_method
        self.data = conversion_method(transitions,
                                      minimal_residence=minimal_residence)

    @property
    def n_jumps(self) -> int:
        """Return total number of jumps."""
        return len(self.data)

    @property
    def n_solo_jumps(self) -> int:
        """Return number of solo jumps."""
        return self.collective().n_solo_jumps

    @property
    def n_floating(self) -> int:
        """Return number of floating species."""
        return len(self.trajectory.species)

    @property
    def solo_fraction(self) -> float:
        """Fraction of solo jumps."""
        return self.n_solo_jumps / self.n_jumps

    @property
    def site_pairs(self) -> list[tuple[str, str]]:
        """Return list of all unique site pairs."""
        labels = self.sites.labels
        site_pairs = product(labels, repeat=2)
        return [pair for pair in site_pairs]

    @property
    def jump_names(self) -> list[str]:
        """Return list of jump names."""
        return ['->'.join(key) for key in self.site_pairs]

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
        lattice = self.trajectory.get_lattice()
        sites = self.sites
        total_time = self.trajectory.total_time

        pdist = lattice.get_all_distances(sites.frac_coords, sites.frac_coords)

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
        return _calculate_transitions_matrix(self.data,
                                             n_sites=self.transitions.n_sites)

    @weak_lru_cache()
    def collective(self, max_dist: float = 1) -> Collective:
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

        trajectory = self.trajectory
        sites = self.transitions.sites

        time_step = trajectory.time_step
        attempt_freq, _ = SimulationMetrics(trajectory).attempt_frequency()

        max_steps = ceil(1.0 / (attempt_freq * time_step))

        return Collective(
            jumps=self,
            sites=sites,
            lattice=trajectory.get_lattice(),
            max_steps=max_steps,
            max_dist=max_dist,
        )

    @weak_lru_cache()
    def activation_energies(self, n_parts: int = 10) -> pd.DataFrame:
        """Calculate activation energies for jumps (UNITS?).

        Parameters
        ----------
        n_parts : 10
            Number of parts to split the data into

        Returns
        -------
        df : pd.DataFrame
            Dataframe with jump activation energies and standard deviations between site pairs.
        """
        trajectory = self.trajectory
        attempt_freq, _ = SimulationMetrics(trajectory).attempt_frequency()

        dct = {}

        temperature = trajectory.metadata['temperature']

        atom_locations_parts = [
            part.atom_locations() for part in self.transitions.split(n_parts)
        ]
        jumps_counter_parts = [
            part.jumps_counter() for part in self.split(n_parts)
        ]
        n_floating = self.n_floating

        for site_pair in self.site_pairs:
            site_start, site_stop = site_pair

            n_jumps = np.array(
                [part[site_pair] for part in jumps_counter_parts])

            part_time = trajectory.total_time / n_parts

            atom_percentage = np.array(
                [part[site_start] for part in atom_locations_parts])

            denom = atom_percentage * n_floating * part_time

            eff_rate = n_jumps / denom

            # For A-A jumps divide by two for a fair comparison of A-A jumps vs. A-B and B-A
            if site_start == site_stop:
                eff_rate /= 2

            e_act_arr = -np.log(eff_rate / attempt_freq) * (
                Boltzmann * temperature) / elementary_charge

            dct[site_start, site_stop] = np.mean(e_act_arr), np.std(e_act_arr,
                                                                    ddof=1)

        df = pd.DataFrame(dct).T
        df.columns = ('energy', 'std')

        return df

    def jumps_counter(self) -> Counter:
        """Calculate number of jumps between sites.

        Returns
        -------
        jumps : dict[tuple[str, str], int]
            Dictionary with number of jumps per sites combination
        """
        labels = self.sites.labels
        jumps = Counter([(labels[i], labels[j]) for _, (
            i, j) in self.data[['start site', 'destination site']].iterrows()])
        return jumps

    def split(self, n_parts: int) -> list[Jumps]:
        """Split the jumps into parts.

        Parameters
        ----------
        n_parts : int
            Number of parts to split the data into

        Returns
        -------
        jumps : list[Jumps]
        """
        parts = self.transitions.split(n_parts)

        return [
            Jumps(part, conversion_method=self.conversion_method)
            for part in parts
        ]

    @weak_lru_cache()
    def rates(self, n_parts: int = 10) -> pd.DataFrame:
        """Calculate jump rates (total jumps / second).

        Returns
        -------
        df : pd.DataFrame
            Dataframe with jump rates and standard deviations between site pairs
        """
        dct = {}

        parts = [part.jumps_counter() for part in self.split(n_parts)]

        for site_pair in self.site_pairs:
            n_jumps = [part[site_pair] for part in parts]

            part_time = self.trajectory.total_time / n_parts
            denom = self.n_floating * part_time

            jump_freq_mean = np.mean(n_jumps) / denom
            jump_freq_std = np.std(n_jumps, ddof=1) / denom

            dct[site_pair] = float(jump_freq_mean), float(jump_freq_std)

        df = pd.DataFrame(dct).T
        df.columns = ('rates', 'std')

        return df

    def plot_jumps_vs_distance(self, **kwargs):
        """See [gemdat.plots.jumps_vs_distance][] for more information."""
        from gemdat import plots
        return plots.jumps_vs_distance(jumps=self, **kwargs)

    def plot_jumps_vs_time(self, **kwargs):
        """See [gemdat.plots.jumps_vs_time][] for more information."""
        from gemdat import plots
        return plots.jumps_vs_time(jumps=self, **kwargs)

    def plot_collective_jumps(self, **kwargs):
        """See [gemdat.plots.collective_jumps][] for more information."""
        from gemdat import plots
        return plots.collective_jumps(jumps=self, **kwargs)

    def plot_jumps_3d(self, **kwargs):
        """See [gemdat.plots.jumps_3d][] for more information."""
        from gemdat import plots
        return plots.jumps_3d(jumps=self, **kwargs)

    def plot_jumps_3d_animation(self, **kwargs):
        """See [gemdat.plots.jumps_3d_animation][] for more information."""
        from gemdat import plots
        return plots.jumps_3d_animation(jumps=self, **kwargs)
