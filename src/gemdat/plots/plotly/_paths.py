from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from gemdat.path import Pathway
from gemdat.volume import Structure, Volume

from ._density import density


def path_on_landscape(
    volume: Volume,
    path: Pathway,
    structure: Structure,
) -> go.Figure:
    """Ploth path over free energy.

    Uses plotly as plotting backend.

    Arguments
    ---------
    volume : Volume
        Input volume to create the landscape
    path : Pathway
        Pathway to plot
    structure : Structure
        Input structure

    Returns
    -------
    fig : go.Figure
        Output as plotly figure
    """

    fig = density(volume.to_probability(), structure)

    path = np.asarray(path.cartesian_path(volume))
    x_path, y_path, z_path = path.T

    # Color path and endpoints differently
    color = ['red' for _ in x_path]
    color[0] = 'blue'
    color[-1] = 'blue'

    fig.add_trace(
        go.Scatter3d(
            x=x_path,
            y=y_path,
            z=z_path,
            mode='markers+lines',
            line={'width': 3},
            marker={
                'size': 6,
                'color': color,
                'symbol': 'circle',
                'opacity': 0.9
            },
            name='Path',
        ))

    return fig
