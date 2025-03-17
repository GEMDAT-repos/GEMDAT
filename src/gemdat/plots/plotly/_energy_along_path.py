from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from pymatgen.core import Structure

if TYPE_CHECKING:
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
        go.Scatter(
            x=np.arange(len(path.energy)),
            y=path.energy,
            name='Optimal path',
            mode='lines',
            line={'width': 3},
        )
    )

    if structure:
        nearest_sites = path.path_over_structure(structure)

        prev_site = nearest_sites[0]
        sections = [(0, prev_site)]

        for i, site in enumerate(nearest_sites):
            if site != prev_site:
                sections.append((i, site))

            prev_site = site

        highlight = True

        for (start, site), (stop, _) in pairwise(sections):
            if highlight:
                fig.add_vrect(
                    x0=start,
                    x1=stop,
                    line_width=0,
                    fillcolor='red',
                    opacity=0.1,
                )

            fig.add_annotation(
                x=(start + stop) / 2,
                y=0.1,
                yref='y domain',
                text=site.label,
                xanchor='center',
                yanchor='middle',
                showarrow=False,
                hovertext=str(site),
            )

            highlight = not highlight

    if other_paths:
        for idx, path in enumerate(other_paths):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(path.energy)),
                    y=path.energy,
                    name=f'Alternative {idx + 1}',
                    mode='lines',
                    line={'width': 1},
                )
            )

    fig.update_layout(title='Pathway', xaxis_title='Steps', yaxis_title='Free energy (eV)')

    return fig
