from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from gemdat.trajectory import Trajectory


def displacement_per_site(*, trajectory: Trajectory) -> plt.Figure:
    """Plot displacement per site.

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
           ylabel='Displacement (Angstrom)')

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
    grouped = defaultdict(list)

    species = trajectory.species

    for sp, distances in zip(species,
                             trajectory.distances_from_base_position()):
        grouped[sp.symbol].append(distances)

    fig, ax = plt.subplots()

    for symbol, distances in grouped.items():
        mean_disp = np.mean(distances, axis=0)
        ax.plot(mean_disp, lw=0.3, label=symbol)

    ax.legend()
    ax.set(title='Displacement per element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    return fig


def msd_per_element(*, trajectory: Trajectory) -> plt.Figure:
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

    for sp in species:
        traj = trajectory.filter(sp.symbol)
        ax.plot(traj.MSD.mean(axis=0), lw=0.5, label=sp.symbol)

    ax.legend()
    ax.set(title='Mean squared displacement per element',
           xlabel='Time step',
           ylabel='MSD (Angstrom$^2$)')

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
    ax.set(title='Histogram of displacements',
           xlabel='Displacement (Angstrom)',
           ylabel='Nr. of atoms')

    return fig
