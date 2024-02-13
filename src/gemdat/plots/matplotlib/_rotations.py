from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def rectilinear_plot(*, grid: np.ndarry) -> plt.Figure:
    """Plot a rectilinear projection of a spherical function.

    Parameters
    ----------
    grid: np.ndarray
        The grid

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    values = grid.T
    phi = np.linspace(0, 360, np.ma.size(values, 0))
    theta = np.linspace(0, 180, np.ma.size(values, 1))

    theta, phi = np.meshgrid(theta, phi)

    fig, ax = plt.subplots(subplot_kw=dict(projection='rectilinear'))
    cs = ax.contourf(phi, theta, values, cmap='viridis')
    ax.set_yticks(np.arange(0, 190, 45))
    ax.set_xticks(np.arange(0, 370, 45))

    ax.set_xlabel(r'azimuthal angle φ $[\degree$]')
    ax.set_ylabel(r'elevation θ $[\degree$]')

    ax.grid(visible=True)
    cbar = fig.colorbar(cs, label='areal probability', format='')

    # Rotate the colorbar label by 180 degrees
    cbar.ax.yaxis.set_label_coords(2.5,
                                   0.5)  # Adjust the position of the label
    cbar.set_label('areal probability', rotation=270, labelpad=15)
    return fig
