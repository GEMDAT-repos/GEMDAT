from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from gemdat import SimulationData, SitesData


def plot_jumps_vs_distance(*,
                           data: SimulationData,
                           sites: SitesData,
                           jump_res: float = 0.1,
                           **kwargs) -> plt.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    sites : SitesData
        Input sites data
    jump_res : float, optional
        Resolution of the bins in Angstrom

    Returns
    -------
    plt.Figure
    """
    lattice = data.lattice
    structure = sites.structure
    pdist = lattice.get_all_distances(structure.frac_coords,
                                      structure.frac_coords)

    bin_max = (1 + pdist.max() // jump_res) * jump_res
    n_bins = int(bin_max / jump_res) + 1
    x = np.linspace(0, bin_max, n_bins)
    counts = np.zeros_like(x)

    bin_idx = np.digitize(pdist, bins=x)
    for idx, n in zip(bin_idx.flatten(), sites.transitions.flatten()):
        counts[idx] += n

    fig, ax = plt.subplots()

    ax.bar(x, counts, width=(jump_res * 0.8))

    ax.set(title='Frequency vs Occurence',
           xlabel='Frequency (Hz)',
           ylabel='Occurrence (a.u.)')

    return fig
