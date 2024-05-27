from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from gemdat.simulation_metrics import SimulationMetrics
from gemdat.trajectory import Trajectory


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

    ax.set(
        title='Histogram of vibrational amplitudes with fitted Gaussian',
        xlabel='Amplitude (Å)',
        ylabel='Occurrence (a.u.)',
    )

    return fig
