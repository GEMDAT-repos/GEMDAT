from __future__ import annotations

import plotly.graph_objects as go
from pymatgen.core import Structure
import numpy as np
from gemdat.path import Pathway


def energy_along_path(
    path: Pathway,
    *,
    structure: Structure | None = None,
    other_paths: list[Pathway] | None = None,
) -> go.Figure:
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
    fig : plotly.graph_objects.Figure
        Output figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=np.arange(len(path.energy)),
                   y=path.energy,
                   name='Optimal path',
                   mode='lines',
                   line={'width': 3}))

    if structure:
        nearest_sites = path.path_over_structure(structure)

        site_xlabel = []
        sitecoord_xlabel = []

        prev = nearest_sites[0]
        for i, site in enumerate(nearest_sites):
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

        for start, stop in zip(non_empty_ticks[::2], non_empty_ticks[1::2]):
            fig.add_vrect(x0=start,
                          x1=stop,
                          line_width=0,
                          fillcolor='red',
                          opacity=0.1)

    if other_paths:
        for idx, path in enumerate(other_paths):
            fig.add_trace(
                go.Scatter(x=np.arange(len(path.energy)),
                           y=path.energy,
                           name=f'Alternative {idx+1}',
                           mode='lines',
                           line={'width': 1}))

    fig.update_layout(title='Pathway',
                      xaxis_title='Steps',
                      yaxis_title='Free energy (eV)')

    return fig
