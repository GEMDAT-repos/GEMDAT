from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.electronic_structure import plotter

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


def plot_jumps_3d(*, data: SimulationData, sites: SitesData,
                  **kwargs) -> plt.Figure:
    """Plot jumps in 3D.

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

    class LabelItems:

        def __init__(self, labels, coords):
            self.labels = labels
            self.coords = coords

        def items(self):
            yield from zip(self.labels, self.coords)

    coords = sites.structure.frac_coords
    lattice = data.structure.lattice

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    site_labels = LabelItems(sites.structure.labels, coords)

    xyz_labels = LabelItems('OABC', [[-0.1, -0.1, -0.1], [1.1, -0.1, -0.1],
                                     [-0.1, 1.1, -0.1], [-0.1, -0.1, 1.1]])

    plotter.plot_lattice_vectors(lattice, ax=ax, linewidth=1)
    plotter.plot_labels(xyz_labels,
                        lattice=lattice,
                        ax=ax,
                        color='green',
                        size=12)
    plotter.plot_points(coords, lattice=lattice, ax=ax)

    for i, j in zip(*np.triu_indices(len(coords), k=1)):
        count = sites.transitions[i, j] + sites.transitions[j, i]
        if count == 0:
            continue

        coord_i = coords[i]
        coord_j = coords[j]

        lw = 1 + np.log(count)

        _, image = lattice.get_distance_and_image(coord_i, coord_j)

        # NOTE: might need to plot `line = [coord_i - image, coord_j]` as well
        line = [coord_i, coord_j + image]

        plotter.plot_path(line,
                          lattice=lattice,
                          ax=ax,
                          color='red',
                          linewidth=lw)

    plotter.plot_labels(site_labels,
                        lattice=lattice,
                        ax=ax,
                        color='black',
                        size=8)

    ax.set(
        title='Jumps between sites',
        xlabel="x' (ang)",
        ylabel="y' (ang)",
        zlabel="z' (ang)",
    )

    ax.set_aspect('equal')  # only auto is supported

    return fig
