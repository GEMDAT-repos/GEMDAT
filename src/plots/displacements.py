import matplotlib.pyplot as plt
import numpy as np
from gemdat.trajectory import Trajectory


def plot_displacement_per_site(*, trajectory: Trajectory,
                               **kwargs) -> plt.Figure:
    """Plot displacement per site.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    **kwargs : dict
        Extra parameters

    Returns
    -------
    fig : plt.Figure
        Output matplotlib figure
    """
    fig, ax = plt.subplots()

    for distances in trajectory.total_distances():
        ax.plot(distances, lw=0.3)

    ax.set(title='Displacement per site',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    return fig


def plot_displacement_per_element(*, trajectory: Trajectory,
                                  **kwargs) -> plt.Figure:
    """Plot displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    **kwargs : dict
        Extra parameters

    Returns
    -------
    fig : plt.Figure
        Output matplotlib figure
    """
    from collections import defaultdict

    grouped = defaultdict(list)

    species = trajectory.species

    for sp, distances in zip(species, trajectory.total_distances()):
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


def plot_displacement_histogram(trajectory: Trajectory,
                                **kwargs) -> plt.Figure:
    """Plot histogram of total displacement of diffusing element at final
    timestep.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    **kwargs : dict
        Extra parameters

    Returns
    -------
    fig : plt.Figure
        Output matplotlib figure
    """
    fig, ax = plt.subplots()
    ax.hist(trajectory.total_distances()[:, -1])
    ax.set(title='Histogram of displacements',
           xlabel='Displacement (Angstrom)',
           ylabel='Nr. of atoms')

    return fig
