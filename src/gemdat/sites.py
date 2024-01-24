from __future__ import annotations

import typing
import warnings
from collections import defaultdict
from itertools import product
from typing import Optional

import numpy as np
from pymatgen.core import Structure

from .simulation_metrics import SimulationMetrics
from .utils import is_lattice_similar

if typing.TYPE_CHECKING:
    from gemdat.trajectory import Trajectory
    from gemdat.transitions import Transitions

NOSITE = -1


class SitesData:

    def __init__(
        self,
        *,
        structure: Structure,
        trajectory: Trajectory,
        floating_specie: str,
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
        site_radius: Optional[float]
            if set it fixes the site_radius instead of determining it
            dynamically
        """
        if not trajectory.constant_lattice:
            raise ValueError(
                'Trajectory must have constant lattice for site analysis.')

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

    def atom_locations(self, transitions: Transitions):
        """Calculate fraction of time atoms spent at a type of site.

        Returns
        -------
        dict[str, float]
            Return dict with the fraction of time atoms spent at a site
        """
        multiplier = self.n_sites / self.n_floating

        compositions_by_label = defaultdict(list)

        for site in transitions.occupancy():
            compositions_by_label[site.label].append(site.composition)

        ret = {}

        for k, v in compositions_by_label.items():
            ret[k] = sum(v) * multiplier

        return ret
