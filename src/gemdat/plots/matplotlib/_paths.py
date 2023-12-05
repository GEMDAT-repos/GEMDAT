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
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(path.energy)), path.energy, marker='o', color='r')
    ax.set(ylabel='Free energy [eV]')
    if path.nearest_structure_label:
        # Remove repeated labels that would bloat the figure
        clean_xlabel = [
            f'{", ".join([f"{coord:.1f}" for coord in path.nearest_structure_coord[i]])}'
            if (path.nearest_structure_coord[i]
                != path.nearest_structure_coord[i - 1]).any() else ''
            for i in range(len(path.sites))
        ]
        non_empty_ticks = [
            i for i, label in enumerate(clean_xlabel) if label != ''
        ]
        ax.set_xticks(non_empty_ticks)
        ax.set_xticklabels([clean_xlabel[i] for i in non_empty_ticks],
                           rotation=45)
        # and add on top the site labels
        clean_xlabel_up = [
            f'{path.nearest_structure_label[i]}' if
            (path.nearest_structure_coord[i]
             != path.nearest_structure_coord[i - 1]).any() else ''
            for i in range(len(path.sites))
        ]
        ax_up = ax.twiny()
        ax_up.set_xlim(ax.get_xlim())
        ax_up.set_xticks(non_empty_ticks)
        ax_up.set_xticklabels([clean_xlabel_up[i] for i in non_empty_ticks],
                              rotation=45)
        ax_up.get_yaxis().set_visible(False)

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
