from __future__ import annotations

import typing
import warnings
from collections import defaultdict
from itertools import product
from typing import Optional

import numpy as np
from pymatgen.core import Structure

from .caching import weak_lru_cache
from .simulation_metrics import SimulationMetrics
from .transitions import Transitions
from .utils import is_lattice_similar

if typing.TYPE_CHECKING:

    from gemdat.trajectory import Trajectory

NOSITE = -1


class SitesData:

    def __init__(
        self,
        *,
        structure: Structure,
        trajectory: Trajectory,
        floating_specie: str,
        n_parts: int = 10,
        site_radius: Optional[float] = None,
    ):
        """Contain sites and jumps data.

        Parameters
        ----------
        structure : pymatgen.core.structure.Structure
            Input structure with known jump sites
        trajectory : Trajectory
            Input trajectory
        floating_specie : str
            Name of the floating or diffusing specie
        n_parts : int, optional
            Number of parts to divide transitions into for statistics
        site_radius: Optional[float]
            if set it fixes the site_radius instead of determining it
            dynamically
        """
        if not trajectory.constant_lattice:
            raise ValueError(
                'Trajectory must have constant lattice for site analysis.')

        self.n_parts = n_parts

        self.floating_specie = floating_specie
        self.structure = structure

        # TODO, these do not belong in this class, move them out
        self.trajectory = trajectory
        self.diff_trajectory = trajectory.filter(self.floating_specie)
        self.metrics = SimulationMetrics(self.diff_trajectory)

        self.warn_if_lattice_not_similar()

    @property
    def site_coords(self) -> np.ndarray:
        """Return fractional coordinates for known sites."""
        return self.structure.frac_coords

    @property
    def site_labels(self) -> list[str]:
        """Return site labels."""
        return self.structure.labels

    @property
    def n_sites(self) -> int:
        """Return number of sites."""
        return len(self.structure)

    @property
    def n_floating(self) -> int:
        """Return number of floating species."""

        return len(self.diff_trajectory.species)

    def warn_if_lattice_not_similar(self):
        """Raise warning if structure and trajectory lattices do not match."""
        this_lattice = self.structure.lattice
        other_lattice = self.trajectory.get_lattice()

        if not is_lattice_similar(other_lattice, this_lattice):
            warnings.warn(f'Lattice mismatch: {this_lattice.parameters} '
                          f'vs. {other_lattice.parameters}')

    @property
    def site_pairs(self) -> list[tuple[str, str]]:
        """Return list of all unique site pairs."""
        labels = self.site_labels
        site_pairs = product(labels, repeat=2)
        return [pair for pair in site_pairs]  # type: ignore

    def occupancy_parts(self,
                        transitions: Transitions) -> list[dict[int, int]]:
        """Return [occupancy arrays][gemdat.transitions.Transitions.occupancy]
        from parts."""
        return [part.occupancy() for part in transitions.split(self.n_parts)]

    def site_occupancy_parts(
            self, transitions: Transitions) -> list[dict[str, float]]:
        """Return [site occupancy][gemdat.sites.SitesData.site_occupancy] dicts
        per part."""
        labels = self.site_labels
        n_steps = len(self.trajectory)

        parts = transitions.split(self.n_parts)

        return [
            _calculate_site_occupancy(occupancy=part.occupancy(),
                                      labels=labels,
                                      n_steps=int(n_steps / self.n_parts))
            for part in parts
        ]

    @weak_lru_cache()
    def atom_locations_parts(
            self, transitions: Transitions) -> list[dict[str, float]]:
        """Return [atom locations][gemdat.sites.SitesData.atom_locations] dicts
        per part."""
        multiplier = self.n_sites / self.n_floating
        return [{
            k: v * multiplier
            for k, v in part.items()
        } for part in self.site_occupancy_parts(transitions)]

    def site_occupancy(self, transitions: Transitions):
        """Calculate percentage occupancy per unique site.

        Returns
        -------
        site_occopancy : dict[str, float]
            Percentage occupancy per unique site
        """
        labels = self.site_labels
        n_steps = len(self.trajectory)
        return _calculate_site_occupancy(occupancy=transitions.occupancy(),
                                         labels=labels,
                                         n_steps=n_steps)

    def atom_locations(self, transitions):
        """Calculate fraction of time atoms spent at a type of site.

        Returns
        -------
        dict[str, float]
            Return dict with the fraction of time atoms spent at a site
        """
        multiplier = self.n_sites / self.n_floating
        return {
            k: v * multiplier
            for k, v in self.site_occupancy(transitions).items()
        }


def _calculate_site_occupancy(
    *,
    occupancy: dict[int, int],
    labels: list[str],
    n_steps: int,
) -> dict[str, float]:
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

    return site_occupancies
