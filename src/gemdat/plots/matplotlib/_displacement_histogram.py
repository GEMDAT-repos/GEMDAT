from __future__ import annotations

import matplotlib.pyplot as plt

from gemdat.trajectory import Trajectory


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
    ax.set(
        title='Displacement per element',
        xlabel='Displacement (Å)',
        ylabel='Nr. of atoms',
    )

    return fig
