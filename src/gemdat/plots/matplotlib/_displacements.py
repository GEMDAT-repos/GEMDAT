from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gemdat.trajectory import Trajectory

from gemdat.plots._shared import _mean_displacements_per_element


def displacement_per_atom(*, trajectory: Trajectory) -> plt.Figure:
    """Plot displacement per atom.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()

    for distances in trajectory.distances_from_base_position():
        ax.plot(distances, lw=0.3)

    ax.set(title='Displacement per site',
           xlabel='Time step',
           ylabel='Displacement (Å)')

    return fig


def displacement_per_element(*, trajectory: Trajectory) -> plt.Figure:
    """Plot displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    displacements = _mean_displacements_per_element(trajectory)

    fig, ax = plt.subplots()

    for symbol, (mean, _) in displacements.items():
        ax.plot(mean, lw=0.3, label=symbol)

    ax.legend()
    ax.set(title='Displacement per element',
           xlabel='Time step',
           ylabel='Displacement (Å)')

    return fig


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


def displacement_histogram(trajectory: Trajectory) -> plt.Figure:
    """Plot histogram of total displacement at final timestep.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()
    ax.hist(trajectory.distances_from_base_position()[:, -1])
    ax.set(title='Displacement per element',
           xlabel='Displacement (Å)',
           ylabel='Nr. of atoms')

    return fig
