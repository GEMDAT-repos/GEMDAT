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

    # Create a colormap to visualize the path
    colormap = plt.get_cmap('viridis')
    normalize = plt.Normalize(0, len(path))

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')

    path_x, path_y, path_z = zip(*path)

    for i in range(len(path) - 1):
        ax.plot(path_x[i:i + 2],
                path_y[i:i + 2],
                path_z[i:i + 2],
                color=colormap(normalize(i)),
                marker='o',
                linestyle='-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Steps')

    return fig
