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


def plot_jumps_vs_time(*,
                       data: SimulationData,
                       sites: SitesData,
                       n_steps: int,
                       binsize: int = 500,
                       **kwargs) -> plt.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    data : SimulationData
        Input simulation data
    sites : SitesData
        Input sites data
    n_steps : int
        Total number of time steps
    binsize : int, optional
        Width of each bin in number of time steps


    Returns
    -------
    plt.Figure
    """
    bins = np.arange(0, n_steps + binsize, binsize)

    fig, ax = plt.subplots()

    ax.hist(sites.all_transitions[:, 4], bins=bins, width=0.8 * binsize)

    ax.set(title='Jumps vs. time',
           xlabel='Time (steps)',
           ylabel='Number of jumps')

    return fig


def plot_collective_jumps(*, data: SimulationData, sites: SitesData,
                          **kwargs) -> plt.Figure:
    """Plot collective jumps per jump-type combination.

    Parameters
    ----------
    data : SimulationData
        Input simulation data
    sites : SitesData
        Input sites data

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots()

    mat = ax.imshow(sites.coll_matrix)

    ticks = range(len(sites.jump_names))

    ax.set_xticks(ticks, labels=sites.jump_names, rotation=90)
    ax.set_yticks(ticks, labels=sites.jump_names)

    fig.colorbar(mat, ax=ax)

    ax.set(title='Cooperative jumps per jump-type combination')

    return fig
