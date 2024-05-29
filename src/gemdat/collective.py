"""This module contains classes for computing which jumps show collective
behaviour."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core import Lattice, Structure

from .caching import weak_lru_cache

if TYPE_CHECKING:
    from .jumps import Jumps


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

    def __init__(
        self,
        jumps: Jumps,
        sites: Structure,
        lattice: Lattice,
        max_steps: int,
        max_dist: float = 1,
    ):
        """Determine number of jumps which could show collective behaviour.

        Parameters
        ----------
        jumps : Jumps
            Input jump events
        sites : pymatgen.core.structure.Structure
            Structure with list of jump sites
        lattice : pymatgen.core.lattice.Lattice
            Input lattice for distance calculations (from simulation data)
        max_steps : int
            Maximum number of time steps which would still mean correlation
        max_dist : float
            Maximum distance for collective motions in Angstrom
        """
        self.jumps = jumps
        self.sites = sites
        self.lattice = lattice
        self.max_steps = max_steps
        self.max_dist = max_dist

        self._compute()

    def _compute(self):
        """Compute number of jumps which could show collective behaviour."""
        sites = self.sites
        lattice = self.lattice
        max_steps = self.max_steps
        max_dist = self.max_dist

        events = self.jumps.data
        events = events.sort_values(['stop time', 'start time'], ignore_index=True)

        coll_jumps = []
        collective = []
        collective_matrix = np.full((len(events), len(events)), False)

        # Compare all pairs
        for i, event_i in events[:-1].iterrows():
            for j, event_j in events[i + 1 :].iterrows():
                if event_j['start time'] - event_i['stop time'] > max_steps:
                    break
                if event_i['start time'] - event_j['stop time'] > max_steps:
                    continue
                if event_i['atom index'] == event_j['atom index']:
                    continue

                a = sites.frac_coords[[event_i['start site'], event_i['destination site']]]
                b = sites.frac_coords[[event_j['start site'], event_j['destination site']]]

                dists = lattice.get_all_distances(a, b)

                if np.any(dists < max_dist):
                    collective.append((event_i, event_j))
                    coll_jumps.append(
                        (
                            (event_i['start site'], event_i['destination site']),
                            (event_j['start site'], event_j['destination site']),
                        )
                    )
                    collective_matrix[i, j] = True
                    collective_matrix[j, i] = True

        # Get solo jumps from the collective matrix
        self.n_solo_jumps = len(events) - np.any(collective_matrix, axis=0).sum()

        self.collective = collective
        self.coll_jumps = coll_jumps
        self.n_coll_jumps = len(events) - self.n_solo_jumps

    @weak_lru_cache()
    def site_pair_count_matrix(self) -> np.ndarray:
        """Collective jumps matrix.

        Returns
        -------
        site_pair_count_matrix : np.ndarray
            Matrix where all types of jumps combinations are counted
        """
        labels = self.sites.labels
        coll_jumps = self.coll_jumps
        site_pairs = self.site_pair_count_matrix_labels()

        site_pair_count_matrix = np.zeros((len(site_pairs), len(site_pairs)), dtype=int)

        for (start_i, stop_i), (start_j, stop_j) in coll_jumps:
            name_start_i = labels[start_i]
            name_stop_i = labels[stop_i]
            name_start_j = labels[start_j]
            name_stop_j = labels[stop_j]

            i = site_pairs.index((name_start_i, name_stop_i))
            j = site_pairs.index((name_start_j, name_stop_j))

            site_pair_count_matrix[i, j] += 1

        return site_pair_count_matrix

    @weak_lru_cache()
    def site_pair_count_matrix_labels(self) -> list:
        labels = self.sites.labels
        return list({(label1, label2) for label1 in labels for label2 in labels})

    @weak_lru_cache()
    def multiple_collective(self) -> tuple[np.ndarray, np.ndarray]:
        """Find jumps that occur collectively multiple times.

        returns collective jumps and their occurence

        Returns
        -------
        multiple_collective : np.ndarray, np.ndarray
            Result of a np.unique on the collective jump pairs
        """
        collective = np.array(
            dtype=[('start', int), ('stop', int)],
            object=[
                [
                    (event_i['start site'], event_i['destination site']),
                    (event_j['start site'], event_j['destination site']),
                ]
                for event_i, event_j in self.collective
            ],
        )
        # Sort jumps so equal jumps are orderd similarily
        collective = np.sort(collective, axis=1)

        # Counts how often collective jumps occur
        jumps, counts = np.unique(collective, axis=0, return_counts=True)

        return jumps, counts
