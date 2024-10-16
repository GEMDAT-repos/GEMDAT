from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rich.progress import track

from ._plot_backend import plot_backend

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from gemdat.transitions import Transitions


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


def _get_states_array(transitions: Transitions, labels: list[str]) -> np.ndarray:
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
        symbol: np.argwhere([sp.symbol == symbol for sp in structure.species]).flatten()
        for symbol in symbols
    }


@dataclass
class RDFData:
    """Data class for storing radial distribution data.

    Parameters
    ----------
    x : np.ndarray
        1D array with x data (bins)
    y : np.ndarray
        1D array with y data (counts)
    symbol : str
        Distance to species with this symbol
    state : str
        State that the floating species is in, e.g.
        the jump that it is making.
    """

    x: np.ndarray
    y: np.ndarray
    symbol: str
    state: str

    @plot_backend
    def plot(self, *, module, **kwargs):
        """See [gemdat.plots.radial_distribution][] for more info."""
        return module.radial_distribution(rdfs=[self], **kwargs)


class RDFCollection(list[RDFData]):
    """Collection to store group of radial distribution data."""

    @plot_backend
    def plot(self, *, module, **kwargs):
        """See [gemdat.plots.radial_distribution][] for more info."""
        return module.radial_distribution(rdfs=self, **kwargs)


def radial_distribution(
    *,
    transitions: Transitions,
    floating_specie: str,
    max_dist: float = 5.0,
    resolution: float = 0.1,
) -> dict[str, RDFCollection]:
    """Calculate and sum RDFs for the floating species in the given sites data.

    Parameters
    ----------
    transitions: Transitions
        Input transitions data
    floating_specie : str
        Name of the floating specie
    max_dist : float, optional
        Max distance for rdf calculation
    resolution : float, optional
        Width of the bins

    Returns
    -------
    rdfs : dict[str, RDFCollection]
        Dictionary with rdf arrays per symbol
    """
    # note: needs trajectory with ALL species
    trajectory = transitions.trajectory
    sites = transitions.sites
    base_structure = trajectory.get_structure(0)
    lattice = trajectory.get_lattice()

    coords = trajectory.positions
    sp_coords = trajectory.filter(floating_specie).positions

    states2str = _get_states(sites.labels)
    states_array = _get_states_array(transitions, sites.labels)
    symbol_indices = _get_symbol_indices(base_structure)

    bins = np.arange(0, max_dist + resolution, resolution)
    length = len(bins) + 1

    rdfs: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(length, dtype=int))

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
                rdfs[state_str, symbol] += np.bincount(rdf_state, minlength=length)

    ret: dict[str, RDFCollection] = {}

    for (state, symbol), values in rdfs.items():
        rdf_data = RDFData(
            x=bins,
            # Drop last element with distance > max_dist
            y=values[:-1],
            symbol=symbol,
            state=state,
        )
        ret.setdefault(state, RDFCollection())
        ret[state].append(rdf_data)

    return ret



import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

def calculate_rdf_parallelized(trajectory, specie_1, specie_2, resolution=1, max_distance=10.0):
    '''Calculate RDFs from specie_1 to specie_2'''
    coords_1 = trajectory.filter(specie_1).coords
    coords_2 = trajectory.filter(specie_2).coords
    lattice = trajectory.get_lattice()

    try:
        num_time_steps, num_atoms, num_dimensions = coords_2.shape
    except ValueError:
        num_time_steps = 1
        num_atoms, num_dimensions = coords_2.shape

    particle_vol = num_atoms / lattice.volume

    def calculate_distances(t):
        return lattice.get_all_distances(coords_1[t, :, :], coords_2[t, :, :])

    all_dists = Parallel(n_jobs=-1)(delayed(calculate_distances)(t) for t in range(num_time_steps))
    distances = np.concatenate([dists[dists != 0].flatten() for dists in all_dists])

    bins = np.arange(0, max_distance + resolution, resolution)
    rdf, _ = np.histogram(distances, bins=bins, density=False)

    norm = np.array([(4 / 3) * np.pi * ((r + resolution) ** 3 - r ** 3) * particle_vol for r in bins[:-1]])
    rdf = np.array([rdf[i] / norm[i] for i in range(len(rdf))])

    return bins, rdf


bins, rdf = calculate_rdf_parallelized(trajectory=trajectory, specie_1='Li', specie_2=['S','Cl'], resolution=0.1, max_distance=10)
plt.plot(bins[:-1], rdf)
