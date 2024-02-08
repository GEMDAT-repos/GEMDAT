from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from gemdat.path import Pathway
from gemdat.volume import Structure, Volume

from ._density import density


def path_on_landscape(
    paths: Pathway | list[Pathway],
    volume: Volume,
    structure: Structure,
) -> go.Figure:
    """Ploth paths over free energy.

    Uses plotly as plotting backend.

    Arguments
    ---------
    paths : Pathway | list[Pathway]
        Pathway object containing the energy along the path, or list of Pathways
    volume : Volume
        Input volume to create the landscape
    structure : Structure
        Input structure

    Returns
    -------
    fig : go.Figure
        Output as plotly figure
    """
    # The first Pathway in paths is assumed to be the optimal one
    if isinstance(paths, Pathway):
        path = paths
    else:
        path = paths[0]

    fig = density(volume.to_probability(), structure)

    x_path, y_path, z_path = np.asarray(path.cartesian_path(volume)).T

    # Plot the optimal path
    fig.add_trace(
        go.Scatter3d(
            x=x_path,
            y=y_path,
            z=z_path,
            mode='markers+lines',
            line={'width': 3},
            marker={
                'size': 6,
                'color': 'teal',
                'symbol': 'circle',
                'opacity': 0.9
            },
            name='Optimal path',
        ))

    # If available, plot the other pathways
    if isinstance(paths, list):
        for idx, path in enumerate(paths[1:]):

            x_path, y_path, z_path = np.asarray(path.cartesian_path(volume)).T

            fig.add_trace(
                go.Scatter3d(
                    x=x_path,
                    y=y_path,
                    z=z_path,
                    mode='markers+lines',
                    line={'width': 3},
                    marker={
                        'size': 5,
                        #'color': color,
                        'symbol': 'circle',
                        'opacity': 0.9
                    },
                    name=f'Alternative {idx+1}',
                ))

    return fig
