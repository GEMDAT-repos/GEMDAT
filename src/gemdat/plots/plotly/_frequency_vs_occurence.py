from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

if TYPE_CHECKING:
    from gemdat.trajectory import Trajectory


def frequency_vs_occurence(*, trajectory: Trajectory) -> go.Figure:
    """Plot attempt frequency vs occurence.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom

    Returns
    -------
    fig : plotly.graph_objects.Figure.Figure
        Output figure
    """
    metrics = trajectory.metrics()
    speed = metrics.speed()

    length = speed.shape[1]
    half_length = length // 2 + 1

    trans = np.fft.fft(speed)

    two_sided = np.abs(trans / length)
    one_sided = two_sided[:, :half_length]

    fig = go.Figure()

    f = trajectory.sampling_frequency * np.arange(half_length) / length

    sum_freqs = np.sum(one_sided, axis=0)
    smoothed = np.convolve(sum_freqs, np.ones(51), 'same') / 51
    fig.add_trace(
        go.Scatter(
            y=smoothed,
            x=f,
            mode='lines',
            line={'width': 3, 'color': 'blue'},
            showlegend=False,
        )
    )

    y_max = np.max(sum_freqs)

    attempt_freq, attempt_freq_std = metrics.attempt_frequency()

    if attempt_freq:
        fig.add_vline(x=attempt_freq, line={'width': 2, 'color': 'red'})
    if attempt_freq and attempt_freq_std:
        fig.add_vline(
            x=attempt_freq + attempt_freq_std,
            line={'width': 2, 'color': 'red', 'dash': 'dash'},
        )
        fig.add_vline(
            x=attempt_freq - attempt_freq_std,
            line={'width': 2, 'color': 'red', 'dash': 'dash'},
        )

    fig.update_layout(
        title='Frequency vs Occurence',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Occurrence (a.u.)',
        xaxis_range=[-0.1e13, 2.5e13],
        yaxis_range=[0, y_max],
        width=600,
        height=500,
    )

    return fig
