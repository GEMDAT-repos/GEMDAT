from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._axes import get_axes

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from gemdat.trajectory import Trajectory


def frequency_vs_occurence(
    *, trajectory: Trajectory, ax: matplotlib.axes.Axes | None = None
) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
    """Plot attempt frequency vs occurence.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom
    ax : matplotlib.axes.Axes | None
        Draw into this axes instead of a new figure; the axes is then returned.

    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes
        Output figure, or `ax` if given.
    """
    metrics = trajectory.metrics()
    speed = metrics.speed()

    length = speed.shape[1]
    half_length = length // 2 + 1

    trans = np.fft.fft(speed)

    two_sided = np.abs(trans / length)
    one_sided = two_sided[:, :half_length]

    fig, ax = get_axes(ax)

    f = trajectory.sampling_frequency * np.arange(half_length) / length

    sum_freqs = np.sum(one_sided, axis=0)
    smoothed = np.convolve(sum_freqs, np.ones(51), 'same') / 51
    ax.plot(f, smoothed, linewidth=3)

    y_max = np.max(sum_freqs)

    attempt_freq, attempt_freq_std = (float(i) for i in metrics.attempt_frequency())

    if attempt_freq:
        ax.vlines([attempt_freq], 0, y_max, colors='red')
    if attempt_freq and attempt_freq_std:
        ax.vlines(
            [attempt_freq + attempt_freq_std, attempt_freq - attempt_freq_std],
            0,
            y_max,
            colors='red',
            linestyles='dashed',
        )

    ax.set(
        title='Frequency vs Occurence',
        xlabel='Frequency (Hz)',
        ylabel='Occurrence (a.u.)',
    )

    ax.set_ylim((0, y_max))
    ax.set_xlim((-0.1e13, 2.5e13))

    return ax if fig is None else fig
