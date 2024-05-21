from __future__ import annotations

import matplotlib.pyplot as plt
from pymatgen.core import Structure

from gemdat.path import Pathway


def energy_along_path(
    path: Pathway,
    *,
    structure: Structure,
    other_paths: list[Pathway] | None = None,
) -> plt.Figure:
    """Plot energy along specified path.

    Parameters
    ----------
    path : Pathway
        Pathway object containing the energy along the path
    structure : Structure
        Structure object to get the site information
    other_paths : Pathway | list[Pathway]
        Optional list of alternative paths to plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(path.energy, marker='o', color='r', label='Optimal path')
    ax.set(ylabel='Free energy (eV)')

    nearest_sites = path.path_over_structure(structure)

    # Create costum labels for the x axis to avoid consecutive repetitions
    site_xlabel = []
    sitecoord_xlabel = []

    prev = nearest_sites[0]
    for i, site in enumerate(nearest_sites):
        # only non repeated labels will get an entry
        if (site.coords != prev.coords).any() or i == 0:
            sitecoord_xlabel.append(', '.join(f'{val:.1f}'
                                              for val in site.coords))
            site_xlabel.append(site.label)
        else:
            sitecoord_xlabel.append('')
            site_xlabel.append('')

        prev = site

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
    if other_paths:
        for idx, path in enumerate(other_paths):
            if path.energy is None:
                raise ValueError('Pathway does not contain energy data')
            ax.plot(range(len(path.energy)),
                    path.energy,
                    label=f'Alternative {idx+1}')
        ax.legend(fontsize=8)

    return fig
