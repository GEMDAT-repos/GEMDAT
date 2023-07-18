from collections import defaultdict

import numpy as np
from gemdat import SimulationData, SitesData


def uniqify_labels(arr, labels):
    unique_labels = list(set(labels))
    mapping = np.array([-1] + [unique_labels.index(label) for label in labels])

    palette = np.arange(len(labels), dtype=int)

    index = np.digitize(arr, palette, right=True)
    return mapping[index]


def get_states(labels):
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
    labels = data.structure.labels

    states2str = get_states(labels)

    atom_sites = uniqify_labels(sites.atom_sites, labels)
    atom_sites_from = uniqify_labels(sites.atom_sites_from, labels)
    atom_sites_to = uniqify_labels(sites.atom_sites_to, labels)

    states_array = (atom_sites * 1e6 + atom_sites_from * 1e3 +
                    atom_sites_to).astype(int)

    lattice = data.lattice

    symbols = data.structure.symbol_set

    symbol_indices = [
        np.argwhere([e.name == symbol
                     for e in data.structure.species]).flatten()
        for symbol in symbols
    ]

    n_symbols = len(symbols)

    bins = np.arange(0, max_dist + resolution, resolution)
    length = len(bins) + 1

    rdfs: dict[tuple[str, str],
               np.ndarray] = defaultdict(lambda: np.zeros(length, dtype=int))

    for i in range(n_steps):
        if i % 1000 == 0:
            print(i)

        all_coords = data.trajectory_coords[equilibration_steps + i]
        diff_coords = diff_coords[i]

        rdf = lattice.get_all_distances(diff_coords, all_coords)

        rdf = np.digitize(rdf, bins, right=True)

        states = np.unique(states_array[i], axis=0)

        for j in range(n_symbols):
            for state in states:
                k_idx = np.argwhere(states_array[i] == state)

                symbol = symbols[j]
                state_str = states2str[state]

                rdf_s = rdf[k_idx, symbol_indices[j]]

                rdfs[state_str, symbol] += np.bincount(rdf_s.flatten(),
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
