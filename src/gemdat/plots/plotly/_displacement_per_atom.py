from __future__ import annotations

import plotly.graph_objects as go

from gemdat.trajectory import Trajectory


def displacement_per_atom(*, trajectory: Trajectory) -> go.Figure:
    """Plot displacement per atom.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    fig = go.Figure()

    distances = [dist for dist in trajectory.distances_from_base_position()]

    for i, distance in enumerate(distances):
        fig.add_trace(
            go.Scatter(y=distance,
                       name=i,
                       mode='lines',
                       line={'width': 1},
                       showlegend=False))

    fig.update_layout(title='Displacement per atom',
                      xaxis_title='Time step',
                      yaxis_title='Displacement (Å)')

    return fig
