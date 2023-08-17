from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rich.progress import track

if TYPE_CHECKING:
    from gemdat import SitesData
    from gemdat.transitions import Transitions
    from pymatgen.core import Structure


def _uniqify_labels(arr, labels: list[str]) -> np.ndarray:
    """Helper function to uniqify labels."""
    unique_labels = list(set(labels))
    mapping = np.array([-1] + [unique_labels.index(label) for label in labels])

    palette = np.arange(len(labels), dtype=int)

    index = np.digitize(arr, palette, right=True)
    return mapping[index]


def _get_states(labels: list[str]) -> dict[int, str]:
    """Helper function to generate a list of states from the labels."""
    unique_labels = list(set(labels))

    states = {}

    r = [-1] + list(range(len(unique_labels)))

    for i in r:
        for j in r:
            for k in r:
                if i != -1:
                    state = '@' + unique_labels[i]
                elif j == -1 or k == -1:
                    state = '~>' + unique_labels[j]
                else:
                    state = unique_labels[j] + '->' + unique_labels[k]

                states[int(i * 1e6 + j * 1e3 + k)] = state

    return states


def _get_states_array(transitions: Transitions,
                      labels: list[str]) -> np.ndarray:
    """Helper function to generate integer array of transition states."""
    states = _uniqify_labels(transitions.states, labels)
    states_prev = _uniqify_labels(transitions.states_prev(), labels)
    states_next = _uniqify_labels(transitions.states_next(), labels)

    states_array = (states * 1e6 + states_prev * 1e3 + states_next).astype(int)

    return states_array


def _get_symbol_indices(structure: Structure) -> dict[str, np.ndarray]:
    """Helper function to generate symbol indices."""
    symbols = structure.symbol_set
    return {
        symbol:
        np.argwhere([sp.symbol == symbol
                     for sp in structure.species]).flatten()
        for symbol in symbols
    }


@dataclass
class RDFData:
    """Container for storing radial distribution data."""
    x: np.ndarray
    y: np.ndarray
    symbol: str
    state: str


def radial_distribution(
        *,
        sites: SitesData,
        max_dist: float = 5.0,
        resolution: float = 0.1) -> dict[str, dict[str, RDFData]]:
    """Calculate and sum RDFs for the floating species in the given sites data.

    Parameters
    ----------
    sites : SitesData
        Input sites data
    max_dist : float, optional
        Max distance for rdf calculation
    resolution : float, optional
        Width of the bins

    Returns
    -------
    rdfs : dict[str, np.ndarray]
        Dictionary with rdf arrays per symbol
    """
    trajectory = sites.trajectory
    structure = trajectory.get_structure(0)
    lattice = trajectory.get_lattice()

    coords = trajectory.positions
    sp_coords = sites.diff_trajectory.positions

    states2str = _get_states(sites.site_labels)
    states_array = _get_states_array(sites.transitions, sites.site_labels)
    symbol_indices = _get_symbol_indices(structure)

    bins = np.arange(0, max_dist + resolution, resolution)
    length = len(bins) + 1

    rdfs: dict[tuple[str, str],
               np.ndarray] = defaultdict(lambda: np.zeros(length, dtype=int))

    n_steps = len(trajectory)

    for i in track(range(n_steps), transient=True):

        t_coords = coords[i]
        t_sp_coords = sp_coords[i]

        dists = lattice.get_all_distances(t_sp_coords, t_coords)

        rdf = np.digitize(dists, bins, right=True)

        states = np.unique(states_array[i], axis=0)

        t_states = states_array[i]

        for state in states:
            k_idx = np.argwhere(t_states == state)
            state_str = states2str[state]

            for symbol, symbol_idx in symbol_indices.items():
                rdf_state = rdf[k_idx, symbol_idx].flatten()
                rdfs[state_str, symbol] += np.bincount(rdf_state,
                                                       minlength=length)

    ret: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    for (state, symbol), values in rdfs.items():
        ret[state][symbol] = RDFData(
            x=bins,
            # Drop last element with distance > max_dist
            y=values[:-1],
            symbol=symbol,
            state=state,
        )

    return ret
