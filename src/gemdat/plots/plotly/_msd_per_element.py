from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

if TYPE_CHECKING:
    from gemdat.trajectory import Trajectory


def hex2rgba(hex_color: str, *, opacity: float = 1) -> str:
    """Convert hex string to rgba."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    return f'rgba({r},{g},{b},{opacity})'


def msd_per_element(*, trajectory: Trajectory) -> go.Figure:
    """Plot mean squared displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    fig = go.Figure()

    time_ps = trajectory.time_step_ps

    species = list(set(trajectory.species))

    for i, sp in enumerate(species):
        color_hex = fig.layout['template']['layout']['colorway'][i]
        color_rgba = hex2rgba(color_hex, opacity=0.3)

        traj = trajectory.filter(sp.symbol)

        msd = traj.mean_squared_displacement()
        msd_mean = np.mean(msd, axis=0)
        msd_std = np.std(msd, axis=0)
        t_values = np.arange(len(msd_mean)) * time_ps

        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=msd_mean + msd_std,
                fillcolor=color_rgba,
                mode='lines',
                line={'width': 0},
                legendgroup=sp.symbol,
                showlegend=False,
                zorder=0,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=msd_mean - msd_std,
                fillcolor=color_rgba,
                mode='none',
                legendgroup=sp.symbol,
                showlegend=False,
                fill='tonexty',
                zorder=0,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=msd_mean,
                name=f'{sp.symbol} mean+std',
                line_color=color_hex,
                mode='lines',
                line={'width': 3},
                legendgroup=sp.symbol,
                zorder=1,
            )
        )

    fig.update_layout(
        title='Mean squared displacement per element',
        xaxis_title='Time lag (ps)',
        yaxis_title=r'MSD (Å<sup>2</sup>)',
    )

    return fig
