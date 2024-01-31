from __future__ import annotations

import typing
from itertools import product
from typing import Optional

from pymatgen.core import Structure

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

    @property
    def site_pairs(self) -> list[tuple[str, str]]:
        """Return list of all unique site pairs."""
        labels = self.structure.labels
        site_pairs = product(labels, repeat=2)
        return [pair for pair in site_pairs]
