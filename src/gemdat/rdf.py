from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rich.progress import track

from ._plot_backend import plot_backend

if TYPE_CHECKING:
    from typing import Collection

    from pymatgen.core import Structure

    from gemdat import Trajectory
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
    label : str
        Distance to species with this symbol label
    state : str
        State that the floating species is in, e.g.
        the jump that it is making.
    """

    x: np.ndarray
    y: np.ndarray
    label: str
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

    states2str = _get_states(sites.labels)  # type: ignore
    states_array = _get_states_array(transitions, sites.labels)  # type: ignore
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
            label=symbol,
            state=state,
        )
        ret.setdefault(state, RDFCollection())
        ret[state].append(rdf_data)

    return ret


def radial_distribution_between_species(
    *,
    trajectory: Trajectory,
    specie_1: str | Collection[str],
    specie_2: str | Collection[str],
    max_dist: float = 5.0,
    resolution: float = 0.1,
) -> RDFData:
    """Calculate RDFs from specie_1 to specie_2.

    Parameters
    ----------
    trajectory: Trajectory
        Input trajectory.
    specie_1: str | list[str]
        Name of specie or list of species
    specie_2: str | list[str]
        Name of specie or list of species
    max_dist: float, optional
        Max distance for rdf calculation
    resolution: float, optional
        Width of the bins

    Returns
    -------
    rdf : RDFData
        RDF data for the given species.
    """
    coords_1 = trajectory.filter(specie_1).coords
    coords_2 = trajectory.filter(specie_2).coords
    lattice = trajectory.get_lattice()

    if coords_2.ndim == 2:
        num_time_steps = 1
        num_atoms, num_dimensions = coords_2.shape
    else:
        num_time_steps, num_atoms, num_dimensions = coords_2.shape

    particle_vol = num_atoms / lattice.volume

    all_dists = np.concatenate(
        [
            lattice.get_all_distances(coords_1[t, :, :], coords_2[t, :, :])
            for t in range(num_time_steps)
        ]
    )
    distances = all_dists.flatten()

    bins = np.arange(0, max_dist + resolution, resolution)
    rdf, _ = np.histogram(distances, bins=bins, density=False)

    def normalize(radius: np.ndarray) -> np.ndarray:
        """Normalize bin to volume."""
        shell = (radius + resolution) ** 3 - radius**3
        return particle_vol * (4 / 3) * np.pi * shell

    norm = normalize(bins)[:-1]
    counts = rdf / norm

    str1 = specie_1 if isinstance(specie_1, str) else '/'.join(specie_1)
    str2 = specie_2 if isinstance(specie_2, str) else '/'.join(specie_2)

    return RDFData(x=bins[:-1], y=counts, label=f'{str1}-{str2}', state='')
