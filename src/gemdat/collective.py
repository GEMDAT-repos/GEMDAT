"""This module contains classes for computing which jumps show collective
behaviour."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure

from .caching import weak_lru_cache
from .transitions import Transitions


class Collective:
    """Data class for collective jumps.

    Attributes
    ----------
    n_solo_jumps : int
        Number of solo jumps
    coll_jumps : list[tuple(int, int)]
        List with start/stop site indices involved in collective jumps
    collective : list[tuple[int, int]]
        List of indices to collective jump events
    """

    def __init__(self,
                 transitions: Transitions,
                 structure: Structure,
                 lattice: Lattice,
                 max_steps: int,
                 max_dist: float = 4.5):
        """Determine number of jumps which could show collective behaviour.

        Parameters
        ----------
        transitions : Transitions
            Input transition events
        structure : pymatgen.core.structure.Structure
            Structure with list of jump sites
        lattice : pymatgen.core.lattice.Lattice
            Input lattice for distance calculations (from simulation data)
        max_steps : int
            Maximum number of time steps which would still mean correlation
        max_dist : float
            Maximum distance for collective motions in Angstrom
        """
        self.transitions = transitions
        self.structure = structure
        self.lattice = lattice
        self.max_steps = max_steps
        self.max_dist = max_dist

        self._compute()

    def _compute(self):
        """Compute number of jumps which could show collective behaviour."""
        structure = self.structure
        lattice = self.lattice
        max_steps = self.max_steps
        max_dist = self.max_dist

        time_col = 4
        events = self.transitions.events
        event_order = events[:, time_col].argsort()

        coll_jumps = []
        collective = []

        n_solo_jumps = 0

        for i, event_i in enumerate(event_order[:-1]):

            for j, event_j in enumerate(event_order[i + 1:], start=1):

                atom_i, start_site_i, stop_site_i, _, stop_time_i = events[
                    event_i]
                atom_j, start_site_j, stop_site_j, _, stop_time_j = events[
                    event_j]

                if stop_time_j - stop_time_i > max_steps:
                    break

                if atom_i == atom_j:
                    n_solo_jumps += 1
                    continue

                a = structure.frac_coords[[start_site_i, stop_site_i]]
                b = structure.frac_coords[[start_site_j, stop_site_j]]

                dists = lattice.get_all_distances(a, b)

                if np.any(dists < max_dist):
                    collective.append((event_i, event_j))
                    coll_jumps.append(((start_site_i, stop_site_i),
                                       (start_site_j, stop_site_j)))

                else:
                    n_solo_jumps += 1

        self.collective = collective
        self.coll_jumps = coll_jumps
        self.n_solo_jumps = n_solo_jumps

    @weak_lru_cache()
    def matrix(self) -> np.ndarray:
        """Calculate collective jumps matrix.

        Returns
        -------
        collective_matrix : np.ndarray
            Matrix where all types of jumps combinations are counted
        """
        labels = self.structure.labels
        coll_jumps = self.coll_jumps

        site_pairs = list({(label1, label2)
                           for label1 in labels
                           for label2 in labels})

        collective_matrix = np.zeros((len(site_pairs), len(site_pairs)),
                                     dtype=int)

        for ((start_i, stop_i), (start_j, stop_j)) in coll_jumps:
            name_start_i = labels[start_i]
            name_stop_i = labels[stop_i]
            name_start_j = labels[start_j]
            name_stop_j = labels[stop_j]

            i = site_pairs.index((name_start_i, name_stop_i))
            j = site_pairs.index((name_start_j, name_stop_j))

            collective_matrix[i, j] += 1

        return collective_matrix

    @weak_lru_cache()
    def multiple_collective(self) -> np.ndarray:
        """Find jumps that occur collectively multiple times.

        Only returns non-unique jumps

        Returns
        -------
        multiple_collective : np.ndarray
            Dictionary with indices of sites between which jumps happen and their counts.
        """
        collective = self.collective
        coll_sorted = np.sort(np.array(collective).flatten())
        difference = np.diff(coll_sorted, prepend=0)
        multiple_collective = coll_sorted[difference == 0]

        return multiple_collective
