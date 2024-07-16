from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from gemdat.plots._shared import hex2rgba

if TYPE_CHECKING:
    from gemdat.orientations import Orientations


def autocorrelation(
    *,
    orientations: Orientations,
    show_traces: bool = True,
    show_shaded: bool = True,
) -> go.Figure:
    """Plot the autocorrelation function of the unit vectors series.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    show_traces : bool
        If True, show traces of individual trajectories
    show_shaded : bool
        If True, show standard deviation as shaded area

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    ac = orientations.autocorrelation()
    ac_std = ac.std(axis=0)
    ac_mean = ac.mean(axis=0)

    time_ps = orientations._time_step * 1e12
    t_values = np.arange(ac_mean.shape[0]) * time_ps

    fig = go.Figure()

    color_hex = fig.layout['template']['layout']['colorway'][0]
    color_rgba = hex2rgba(color_hex, opacity=0.3)

    fig.add_trace(
        go.Scatter(
            x=t_values,
            y=ac_mean,
            line_color=color_hex,
            name='FFT Autocorrelation',
            mode='lines',
            line={'width': 3},
            legendgroup='autocorr',
            zorder=10,
        )
    )

    if show_shaded:
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=ac_mean + ac_std,
                fillcolor=color_rgba,
                mode='lines',
                line={'width': 0},
                legendgroup='autocorr',
                showlegend=False,
                zorder=0,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_values,
                y=ac_mean - ac_std,
                fillcolor=color_rgba,
                mode='none',
                legendgroup='autocorr',
                showlegend=False,
                fill='tonexty',
                zorder=0,
            )
        )

    if show_traces:
        for i, trace in enumerate(ac):
            fig.add_trace(
                go.Scatter(
                    x=t_values,
                    y=trace,
                    name=i,
                    mode='lines',
                    line={'width': 0.25},
                    showlegend=False,
                    zorder=5,
                )
            )

    fig.update_layout(
        title='FFT Autocorrelation',
        xaxis_title='Time lag (ps)',
        yaxis_title='mean + std',
    )

    return fig
