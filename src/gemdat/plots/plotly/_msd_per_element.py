from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from gemdat.trajectory import Trajectory


def msd_per_element(*, trajectory: Trajectory) -> go.Figure:
    """Plot mean squared displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    fig = go.Figure()

    species = list(set(trajectory.species))

    # Since we want to plot in picosecond, we convert the time units
    time_ps = trajectory.time_step * 1e12

    for sp in species:
        traj = trajectory.filter(sp.symbol)

        msd = traj.mean_squared_displacement()
        msd_mean = np.mean(msd, axis=0)
        msd_std = np.std(msd, axis=0)
        t_values = np.arange(len(msd_mean)) * time_ps

        fig.add_trace(
            go.Scatter(x=t_values,
                       y=msd_mean,
                       error_y=dict(type='data',
                                    array=msd_std,
                                    width=0.1,
                                    thickness=0.1),
                       name=sp.symbol,
                       mode='lines',
                       line={'width': 3},
                       legendgroup=sp.symbol))

    fig.update_layout(title='Mean squared displacement per element',
                      xaxis_title='Time lag (ps)',
                      yaxis_title=r'MSD (Å<sup>2</sup>)')

    return fig
