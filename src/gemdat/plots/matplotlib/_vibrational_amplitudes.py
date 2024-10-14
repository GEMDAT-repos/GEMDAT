from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .._shared import _get_vibrational_amplitudes_hist

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat.trajectory import Trajectory


def vibrational_amplitudes(
    *, trajectory: Trajectory, bins: int = 50, n_parts: int = 1
) -> matplotlib.figure.Figure:
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
    metrics = trajectory.metrics()

    trajectories = trajectory.split(n_parts)

    hist = _get_vibrational_amplitudes_hist(trajectories=trajectories, bins=bins)
    fig, ax = plt.subplots()

    plt.bar(hist.amplitudes + hist.offset, hist.counts, width=hist.width, yerr=hist.std)

    x = np.linspace(hist.min_amp, hist.max_amp, 100)
    y_gauss = stats.norm.pdf(x, 0, metrics.vibration_amplitude())
    ax.plot(x, y_gauss, 'r')

    ax.set(
        title='Histogram of vibrational amplitudes with fitted Gaussian',
        xlabel='Amplitude (Å)',
        ylabel='Occurrence (a.u.)',
    )

    return fig
