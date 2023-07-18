import numpy as np
from gemdat import SimulationData, SitesData


def calculate_rdfs(*,
                   data: SimulationData,
                   sites: SitesData,
                   diff_coords: np.ndarray,
                   n_steps: int,
                   equilibration_steps: int,
                   max_dist: float = 5.0,
                   resolution: float = 0.1) -> dict[str, np.ndarray]:
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
    lattice = data.lattice

    symbols = data.structure.symbol_set

    symbol_indices = [
        np.argwhere([e.name == symbol
                     for e in data.structure.species]).flatten()
        for symbol in symbols
    ]

    n_symbols = len(symbols)

    bin_width = 0.1
    bins = np.arange(0, max_dist + bin_width, bin_width)

    rdfs = [np.zeros(len(bins) + 1, dtype=int) for symbol in symbols]

    for i in range(n_steps):
        all_coords = data.trajectory_coords[equilibration_steps + i]
        diff_coords = diff_coords[i]

        rdf = lattice.get_all_distances(diff_coords, all_coords)

        rdf = np.digitize(rdf, bins, right=True)

        for i in range(n_symbols):
            rdf_s = rdf[:, symbol_indices[i]]
            rdfs[i] += np.bincount(rdf_s.flatten())

    # Drop last element with distance > max_dist
    rdfs = [rdf[:-1] for rdf in rdfs]

    return dict(zip(symbols, rdfs))


def plot_rdf(rdfs: dict[str, np.ndarray]):
    """Plot radial distribution function.

    Parameters
    ----------
    rdfs : dict[str, np.ndarray]
        Dictionary with rdf array per symbol
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for symbol, rdf in rdfs.items():
        ax.plot(rdf, label=symbol)

    ax.legend()
    ax.set(title='Radial distribution function',
           xlabel='Distance (Ang)',
           ylabel='Counts')
