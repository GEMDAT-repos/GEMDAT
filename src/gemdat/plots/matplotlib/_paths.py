from __future__ import annotations

import matplotlib.pyplot as plt
from pymatgen.core import Structure

from gemdat.path import Pathway
from gemdat.volume import Volume


def energy_along_path(
    *,
    paths: Pathway | list[Pathway],
    structure: Structure,
    volume: Volume,
) -> plt.Figure:
    """Plot energy along specified path.

    Parameters
    ----------
    paths : Pathway | list[Pathway]
        Pathway object containing the energy along the path, or list of Pathways
    structure : Structure
        Structure object to get the site information
    volume : Volume
        Volume object to get the site information

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    # The first Pathway in paths is assumed to be the optimal one
    if isinstance(paths, list):
        path = paths[0]
    else:
        path = paths

    if path.energy is None:
        raise ValueError('Pathway does not contain energy data')
    if path.sites is None:
        raise ValueError('Pathway does not contain site data')

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(path.energy, marker='o', color='r', label='Optimal path')
    ax.set(ylabel='Free energy [eV]')

    nearest_structure_label, nearest_structure_coord = path.path_over_structure(
        structure, volume)

    # Create costum labels for the x axis to avoid consecutive repetitions
    site_xlabel = []
    sitecoord_xlabel = []
    for i in range(len(path.sites)):
        coord = nearest_structure_coord[i]
        # only non repeated labels will get an entry
        if (coord != nearest_structure_coord[i - 1]).any() or i == 0:
            sitecoord_xlabel.append(', '.join([f'{val:.1f}' for val in coord]))
            site_xlabel.append(f'{nearest_structure_label[i]}')
        else:
            sitecoord_xlabel.append('')
            site_xlabel.append('')
    non_empty_ticks = [
        i for i, label in enumerate(sitecoord_xlabel) if label != ''
    ]
    extra_ticks = non_empty_ticks.copy()
    extra_ticks.append(ax.get_xlim()[1])
    centered_ticks = [(extra_ticks[i] + extra_ticks[i + 1]) / 2
                      for i in range(len(extra_ticks) - 1)]
    ax.set_xticks(centered_ticks)
    ax.set_xticklabels([sitecoord_xlabel[i] for i in non_empty_ticks],
                       rotation=45)
    # Change background color alternatively for different sites
    for i in range(0, len(non_empty_ticks), 2):
        if i + 1 < len(non_empty_ticks):
            ax.axvspan(non_empty_ticks[i],
                       non_empty_ticks[i + 1],
                       facecolor='lightgray',
                       edgecolor='none')
        else:
            ax.axvspan(non_empty_ticks[i],
                       max(non_empty_ticks),
                       facecolor='lightgray',
                       edgecolor='none')
    # and add on top the site labels
    ax_up = ax.twiny()
    ax_up.set_xlim(ax.get_xlim())
    ax_up.set_xticks(centered_ticks)
    ax_up.set_xticklabels([site_xlabel[i] for i in non_empty_ticks],
                          rotation=45)
    ax_up.get_yaxis().set_visible(False)

    # If available, plot the other pathways
    if isinstance(paths, list):
        for idx, path in enumerate(paths[1:]):
            if path.energy is None:
                raise ValueError('Pathway does not contain energy data')
            ax.plot(range(len(path.energy)),
                    path.energy,
                    label=f'Alternative {idx+1}')
        ax.legend(fontsize=8)

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
    if path.energy is None:
        raise ValueError('Pathway does not contain energy data')
    if path.sites is None:
        raise ValueError('Pathway does not contain site data')

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
