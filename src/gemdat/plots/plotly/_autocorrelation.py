from __future__ import annotations
import plotly.graph_objects as go
from gemdat.orientations import Orientations
import numpy as np


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

    if show_shaded:
        error_y = {
            'type': 'data',
            'array': ac_std,
            'width': 0.1,
            'thickness': 0.1
        }
    else:
        error_y = None

    fig.add_trace(
        go.Scatter(x=t_values,
                   y=ac_mean,
                   error_y=error_y,
                   name='FFT Autocorrelation',
                   mode='lines',
                   line={'width': 3},
                   legendgroup='autocorr'))

    if show_traces:
        for i, trace in enumerate(ac):
            fig.add_trace(
                go.Scatter(x=t_values,
                           y=trace,
                           name=i,
                           mode='lines',
                           line={'width': 0.25},
                           showlegend=False))

    fig.update_layout(title='FFT Autocorrelation',
                      xaxis_title='Time lag (ps)',
                      yaxis_title='mean + std')

    return fig
