from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from gemdat.simulation_metrics import SimulationMetrics
from gemdat.trajectory import Trajectory


def frequency_vs_occurence(*, trajectory: Trajectory) -> plt.Figure:
    """Plot attempt frequency vs occurence.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    metrics = SimulationMetrics(trajectory)
    speed = metrics.speed()

    length = speed.shape[1]
    half_length = length // 2 + 1

    trans = np.fft.fft(speed)

    two_sided = np.abs(trans / length)
    one_sided = two_sided[:, :half_length]

    fig, ax = plt.subplots()

    f = trajectory.sampling_frequency * np.arange(half_length) / length

    sum_freqs = np.sum(one_sided, axis=0)
    smoothed = np.convolve(sum_freqs, np.ones(51), 'same') / 51
    ax.plot(f, smoothed, linewidth=3)

    y_max = np.max(sum_freqs)

    attempt_freq, attempt_freq_std = metrics.attempt_frequency()

    if attempt_freq:
        ax.vlines([attempt_freq], 0, y_max, colors='red')
    if attempt_freq and attempt_freq_std:
        ax.vlines(
            [attempt_freq + attempt_freq_std, attempt_freq - attempt_freq_std],
            0,
            y_max,
            colors='red',
            linestyles='dashed')

    ax.set(title='Frequency vs Occurence',
           xlabel='Frequency (Hz)',
           ylabel='Occurrence (a.u.)')

    ax.set_ylim([0, y_max])
    ax.set_xlim([-0.1e13, 2.5e13])

    return fig


def vibrational_amplitudes(*, trajectory: Trajectory) -> plt.Figure:
    """Plot histogram of vibrational amplitudes with fitted Gaussian.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    metrics = SimulationMetrics(trajectory)

    fig, ax = plt.subplots()
    ax.hist(metrics.amplitudes(), bins=100, density=True)

    x = np.linspace(-2, 2, 100)
    y_gauss = stats.norm.pdf(x, 0, metrics.vibration_amplitude())
    ax.plot(x, y_gauss, 'r')

    ax.set(title='Histogram of vibrational amplitudes with fitted Gaussian',
           xlabel='Amplitude (Angstrom)',
           ylabel='Occurrence (a.u.)')

    return fig


def vibrational_amplitudes2(*,
                            trajectory: Trajectory,
                            bins: int = 50,
                            n_parts: int = 1) -> go.Figure:
    """Plot histogram of vibrational amplitudes with fitted Gaussian.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom
    n_parts : int
        Number of parts for error analysis

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    trajectories = trajectory.split(n_parts)
    single_metrics = SimulationMetrics(trajectory)
    metrics = [
        SimulationMetrics(trajectory).amplitudes()
        for trajectory in trajectories
    ]

    max_amp = max(max(metric) for metric in metrics))
    min_amp = min(min(metric) for metric in metrics))
    
    max_amp = max(abs(min_amp), max_amp)
    min_amp = -max_val

    data = []

    for metric in metrics:
        data.append(
            np.histogram(metric,
                         bins=bins,
                         range=(minamp, maxamp),
                         density=True)[0])

    df = pd.DataFrame(data=data)

    # offset to middle of bar
    offset = (maxamp - minamp) / (bins * 2)

    columns = np.linspace(minamp + offset,
                          maxamp + offset,
                          bins,
                          endpoint=False)

    mean = [df[col].mean() for col in df.columns]
    std = [df[col].std() for col in df.columns]

    df = pd.DataFrame(data=zip(columns, mean, std),
                      columns=['amplitude', 'count', 'std'])

    if n_parts == 1:
        fig = px.bar(df, x='amplitude', y='count')
    else:
        fig = px.bar(df, x='amplitude', y='count', error_y='std')

    x = np.linspace(minamp, maxamp, 100)
    y_gauss = stats.norm.pdf(x, 0, single_metrics.vibration_amplitude())
    fig.add_trace(go.Scatter(x=x, y=y_gauss, name='Fitted Gaussian'))

    fig.update_layout(
        title='Histogram of vibrational amplitudes with fitted Gaussian',
        xaxis_title='Amplitude (Ångstrom)',
        yaxis_title='Occurrence (a.u.)')

    return fig
