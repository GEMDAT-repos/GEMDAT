from __future__ import annotations

import matplotlib.pyplot as plt


def energy_along_path(*, energy_path: list) -> plt.Figure:
    """Plot energy along specified path.

    Parameters
    ----------
    energy_path : list
        Energy along the path

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()

    ax.plot(range(len(energy_path)), energy_path)
    ax.set(xlabel='Step', ylabel='Energy')

    return fig


def path_on_grid(*, path: list) -> plt.Figure:
    """Plot path in 3d.

    Parameters
    ----------
    path : list
        List of the coordinates that define the path

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')

    path_x, path_y, path_z = zip(*path)

    ax.plot(path_x, path_y, path_z, marker='o', linestyle='-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return fig
