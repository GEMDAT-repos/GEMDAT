from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gemdat.trajectory import Trajectory


def msd_per_element(
    *,
    trajectory: Trajectory,
) -> plt.Figure:
    """Plot mean squared displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    species = list(set(trajectory.species))

    fig, ax = plt.subplots()

    # Since we want to plot in picosecond, we convert the time units
    time_ps = trajectory.time_step * 1e12

    for sp in species:
        traj = trajectory.filter(sp.symbol)
        msd = traj.mean_squared_displacement()
        msd_mean = np.mean(msd, axis=0)
        msd_std = np.std(msd, axis=0)
        t_values = np.arange(len(msd_mean)) * time_ps
        ax.plot(t_values, msd_mean, lw=0.5, label=sp.symbol)
        last_color = ax.lines[-1].get_color()
        ax.fill_between(t_values,
                        msd_mean - msd_std,
                        msd_mean + msd_std,
                        color=last_color,
                        alpha=0.2)

    ax.legend()
    ax.set(title='Mean squared displacement per element',
           xlabel='Time lag [ps]',
           ylabel='MSD (Ångstrom$^2$)')

    return fig
