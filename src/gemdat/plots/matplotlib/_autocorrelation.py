from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat.orientations import Orientations


def autocorrelation(
    *,
    orientations: Orientations,
    show_traces: bool = True,
    show_shaded: bool = True,
) -> matplotlib.figure.Figure:
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
    fig : matplotlib.figure.Figure
        Output figure
    """
    ac = orientations.autocorrelation()
    ac_std = ac.std(axis=0)
    ac_mean = ac.mean(axis=0)

    # Since we want to plot in picosecond, we convert the time units
    time_ps = orientations._time_step * 1e12
    tgrid = np.arange(ac_mean.shape[0]) * time_ps

    fig, ax = plt.subplots()

    ax.plot(tgrid, ac_mean, label='FFT autocorrelation')

    last_color = ax.lines[-1].get_color()

    if show_traces:
        for i, ac_i in enumerate(ac):
            label = 'Trajectories' if (i == 0) else None
            ax.plot(tgrid, ac_i, lw=0.1, c=last_color, label=label)

    if show_shaded:
        ax.fill_between(
            tgrid,
            ac_mean - ac_std,
            ac_mean + ac_std,
            alpha=0.2,
            label='Standard deviation',
        )

    ax.set_xlabel('Time lag (ps)')
    ax.set_ylabel('Autocorrelation')
    ax.legend()

    return fig
