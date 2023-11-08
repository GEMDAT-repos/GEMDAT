from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from gemdat.volume import Structure, Volume

from ._density import density


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

    ax.plot(range(len(energy_path)), energy_path, marker='o', color='r')
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


def path_on_landscape(
    vol: Volume,
    path: np.ndarray,
    structure: Structure,
) -> go.Figure:
    """Ploth path over free energy.

    Uses plotly as plotting backend.

    Arguments
    ---------
    vol : Volume
        Input volume to create the landscape
    path : np.ndarray
        Path to plot
    structure : Structure
        Input structure

    Returns
    -------
    fig : go.Figure
        Output as plotly figure
    """

    fig = density(vol, structure, as_probability=True)

    x_path, y_path, z_path = zip(*path)
    x_path = np.asarray(x_path) * vol.resolution
    y_path = np.asarray(y_path) * vol.resolution
    z_path = np.asarray(z_path) * vol.resolution

    fig.add_trace(
        go.Scatter3d(
            x=x_path,
            y=y_path,
            z=z_path,
            mode='markers+lines',
            line=dict(width=3),
            marker=dict(size=6,
                        color=[
                            'blue' if i == 0 or i == len(x_path) - 1 else 'red'
                            for i in range(len(x_path))
                        ],
                        symbol='circle',
                        opacity=0.9),
            name='Path',
        ))

    return fig
