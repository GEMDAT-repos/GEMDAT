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
        raise NotImplementedError

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
