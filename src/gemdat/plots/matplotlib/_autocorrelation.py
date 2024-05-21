from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gemdat.orientations import (
    Orientations, )


def autocorrelation(
    *,
    orientations: Orientations,
) -> plt.Figure:
    """Plot the autocorrelation function of the unit vectors series.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    ac = orientations.autocorrelation()
    ac_std = ac.std(axis=0)
    ac_mean = ac.mean(axis=0)

    # Since we want to plot in picosecond, we convert the time units
    time_ps = orientations._time_step * 1e12
    tgrid = np.arange(ac_mean.shape[0]) * time_ps

    # and now we can plot the autocorrelation function
    fig, ax = plt.subplots()

    ax.plot(tgrid, ac_mean, label='FFT-Autocorrelation')
    ax.fill_between(tgrid, ac_mean - ac_std, ac_mean + ac_std, alpha=0.2)
    ax.set_xlabel('Time lag [ps]')
    ax.set_ylabel('Autocorrelation')

    return fig
