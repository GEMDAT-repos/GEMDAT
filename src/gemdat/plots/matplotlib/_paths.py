from __future__ import annotations

import matplotlib.pyplot as plt

from gemdat.path import Pathway


def energy_along_path(*, path: Pathway) -> plt.Figure:
    """Plot energy along specified path.

    Parameters
    ----------
    path : Pathway
        Pathway object containing the energy along the path

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()

    ax.plot(range(len(path.energy)), path.energy, marker='o', color='r')
    ax.set(xlabel='Step', ylabel='Energy')

    return fig


def path_on_grid(*, path: Pathway) -> plt.Figure:
    """Plot the 3d coordinates of the points that define a path.

    Parameters
    ----------
    path : Pathway
        Pathway to plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    # Create a colormap to visualize the path
    colormap = plt.get_cmap()
    normalize = plt.Normalize(0, len(path.energy))

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')

    path_x, path_y, path_z = zip(*path.sites)

    for i in range(len(path.energy) - 1):
        ax.plot(path_x[i:i + 1],
                path_y[i:i + 1],
                path_z[i:i + 1],
                color=colormap(normalize(i)),
                marker='o',
                linestyle='-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Steps')

    return fig
