from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from pymatgen.core import Structure

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat.path import Pathway


def energy_along_path(
    path: Pathway,
    *,
    structure: Structure | None = None,
    other_paths: list[Pathway] | None = None,
) -> matplotlib.figure.Figure:
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
    ax.set_xlabel('Step')
    ax.set_ylabel('Free energy (eV)')

    if structure:
        nearest_sites = path.path_over_structure(structure)

        site_xlabel = []
        sitecoord_xlabel = []

        prev = nearest_sites[0]
        for i, site in enumerate(nearest_sites):
            if (site.coords != prev.coords).any() or i == 0:
                sitecoord_xlabel.append(', '.join(f'{val:.1f}' for val in site.coords))
                site_xlabel.append(site.label)
            else:
                sitecoord_xlabel.append('')
                site_xlabel.append('')

            prev = site

        non_empty_ticks = [i for i, label in enumerate(sitecoord_xlabel) if label != '']

        extra_ticks = non_empty_ticks.copy()
        extra_ticks.append(ax.get_xlim()[1])  # type: ignore
        centered_ticks = [
            (extra_ticks[i] + extra_ticks[i + 1]) / 2 for i in range(len(extra_ticks) - 1)
        ]

        ax.set_xticks(centered_ticks)
        ax.set_xticklabels([sitecoord_xlabel[i] for i in non_empty_ticks], rotation=45)

        for start, stop in zip(non_empty_ticks[::2], non_empty_ticks[1::2]):
            ax.axvspan(start, stop, facecolor='lightgray', edgecolor='none')

        ax_up = ax.twiny()
        ax_up.set_xlim(ax.get_xlim())
        ax_up.set_xticks(centered_ticks)
        ax_up.set_xticklabels([site_xlabel[i] for i in non_empty_ticks], rotation=45)
        ax_up.get_yaxis().set_visible(False)

    if other_paths:
        for idx, path in enumerate(other_paths):
            if path.energy is None:
                raise ValueError('Pathway does not contain energy data')
            ax.plot(range(len(path.energy)), path.energy, label=f'Alternative {idx + 1}')

        ax.legend(fontsize=8)

    return fig
