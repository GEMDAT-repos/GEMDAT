from __future__ import annotations

import plotly.graph_objects as go

from gemdat.trajectory import Trajectory
from gemdat.plots._shared import _mean_displacements_per_element


def displacement_per_element(*, trajectory: Trajectory) -> go.Figure:
    """Plot displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    displacements = _mean_displacements_per_element(trajectory)

    fig = go.Figure()

    for symbol, (mean, std) in displacements.items():
        fig.add_trace(
            go.Scatter(y=mean,
                       name=symbol + ' + std',
                       mode='lines',
                       line={'width': 3},
                       legendgroup=symbol))
        fig.add_trace(
            go.Scatter(y=mean + std,
                       name=symbol + ' + std',
                       mode='lines',
                       line={'width': 0},
                       legendgroup=symbol,
                       showlegend=False))
        fig.add_trace(
            go.Scatter(y=mean - std,
                       name=symbol + ' + std',
                       mode='lines',
                       line={'width': 0},
                       legendgroup=symbol,
                       showlegend=False,
                       fill='tonexty'))

    fig.update_layout(title='Displacement per element',
                      xaxis_title='Time step',
                      yaxis_title='Displacement (Å)')

    return fig
