from collections import defaultdict

import numpy as np
from gemdat import SimulationData, SitesData
from pymatgen.core import Structure
from rich.progress import track


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


def _get_states_array(sites: SimulationData, labels: list[str]) -> np.ndarray:
    """Helper function to generate integer array of transition states."""
    atom_sites = _uniqify_labels(sites.atom_sites, labels)
    atom_sites_from = _uniqify_labels(sites.atom_sites_from, labels)
    atom_sites_to = _uniqify_labels(sites.atom_sites_to, labels)

    states_array = (atom_sites * 1e6 + atom_sites_from * 1e3 +
                    atom_sites_to).astype(int)

    return states_array


def _get_symbol_indices(structure: Structure) -> dict[str, np.ndarray]:
    """Helper function to generate symbol indices."""
    symbols = structure.symbol_set
    return {
        symbol:
        np.argwhere([e.name == symbol for e in structure.species]).flatten()
        for symbol in symbols
    }


def calculate_rdfs(
        *,
        data: SimulationData,
        sites: SitesData,
        diff_coords: np.ndarray,
        n_steps: int,
        equilibration_steps: int,
        max_dist: float = 5.0,
        resolution: float = 0.1) -> dict[str, dict[str, np.ndarray]]:
    """
    Parameters
    ----------
    data : SimulationData
        Input simulation data
    sites : SitesData
        Input sites data
    diff_coords : np.ndarray
        Input coordinates for diffusing element (extras)
    n_steps : int
        Total number of simulation steps (extras)
    equilibration_steps : int
        Number of equilibration steps (extras)
    max_dist : float, optional
        Max distance for rdf calculation
    resolution : float, optional
        Width of the bins

    Returns
    -------
    rdfs : dict[str, np.ndarray]
        Dictionary with rdf arrays per symbol
    """
    structure = data.structure
    lattice = structure.lattice

    coords = data.trajectory_coords[equilibration_steps:]

    states2str = _get_states(sites.structure.labels)
    states_array = _get_states_array(sites, sites.structure.labels)
    symbol_indices = _get_symbol_indices(structure)

    bins = np.arange(0, max_dist + resolution, resolution)
    length = len(bins) + 1

    rdfs: dict[tuple[str, str],
               np.ndarray] = defaultdict(lambda: np.zeros(length, dtype=int))

    for i in track(range(n_steps), transient=True):

        t_coords = coords[i]
        t_diff_coords = diff_coords[i]

        dists = lattice.get_all_distances(t_diff_coords, t_coords)

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

    new_rdf: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    for (k0, k1), v in rdfs.items():
        # Drop last element with distance > max_dist
        new_rdf[k0][k1] = v[:-1]

    return new_rdf


def plot_rdf(rdfs: dict[str, np.ndarray], name: str | None = None):
    """Plot radial distribution function.

    Parameters
    ----------
    rdfs : dict[str, np.ndarray]
        Dictionary with rdf array per symbol
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for symbol, rdf in rdfs.items():
        ax.plot(rdf[:-1], label=symbol)

    suffix = f' ({name})' if name else ''

    ax.legend()
    ax.set(title=f'Radial distribution function{suffix}',
           xlabel='Distance (Ang)',
           ylabel='Counts')
