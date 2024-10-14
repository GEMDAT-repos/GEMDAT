from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from gemdat.trajectory import Trajectory


def vibrational_amplitudes(
    *, trajectory: Trajectory, bins: int = 50, n_parts: int = 1
) -> plt.Figure:
    """Plot histogram of vibrational amplitudes with fitted Gaussian.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom
    bins : int
        Number of bins for the histogram
    n_parts : int
        Number of parts for error analysis

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    trajectories = trajectory.split(n_parts)
    single_metrics = trajectory.metrics()
    metrics = [trajectory.metrics().amplitudes() for trajectory in trajectories]

    max_amp = max(max(metric) for metric in metrics)
    min_amp = min(min(metric) for metric in metrics)

    max_amp = max(abs(min_amp), max_amp)
    min_amp = -max_amp

    data = []

    for metric in metrics:
        data.append(np.histogram(metric, bins=bins, range=(min_amp, max_amp), density=True)[0])

    # offset to middle of bar
    offset = (max_amp - min_amp) / (bins * 2)

    columns = np.linspace(min_amp + offset, max_amp + offset, bins, endpoint=False)

    mean = [col.mean() for col in columns]
    std = [col.std() for col in columns]

    fig, ax = plt.subplots()
    # hist goes here
    # ax.hist(metrics.amplitudes(), bins=bins, density=True)

    plt.hist(columns, columns, weights=mean)

    x = np.linspace(-2, 2, 100)
    y_gauss = stats.norm.pdf(x, 0, single_metrics.vibration_amplitude())
    ax.plot(x, y_gauss, 'r')

    ax.set(
        title='Histogram of vibrational amplitudes with fitted Gaussian',
        xlabel='Amplitude (Å)',
        ylabel='Occurrence (a.u.)',
    )

    return fig
