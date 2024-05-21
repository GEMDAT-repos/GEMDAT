from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gemdat.orientations import (
    Orientations,
    calculate_spherical_areas,
)


def rectilinear(*,
                orientations: Orientations,
                shape: tuple[int, int] = (90, 360),
                normalize_histo: bool = True) -> plt.Figure:
    """Plot a rectilinear projection of a spherical function. This function
    uses the transformed trajectory.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    shape : tuple
        The shape of the spherical sector in which the trajectory is plotted
    normalize_histo : bool, optional
        If True, normalize the histogram by the area of the bins, by default True

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    # Convert the vectors to spherical coordinates
    az, el, _ = orientations.vectors_spherical.T
    az = az.flatten()
    el = el.flatten()

    hist, xedges, yedges = np.histogram2d(el, az, shape)

    if normalize_histo:
        # Normalize by the area of the bins
        areas = calculate_spherical_areas(shape)
        hist = np.divide(hist, areas)
        # Drop the bins at the poles where normalization is not possible
        hist = hist[1:-1, :]

    values = hist.T
    axis_phi, axis_theta = values.shape

    phi = np.linspace(0, 360, axis_phi)
    theta = np.linspace(0, 180, axis_theta)

    theta, phi = np.meshgrid(theta, phi)

    fig, ax = plt.subplots(subplot_kw=dict(projection='rectilinear'))
    cs = ax.contourf(phi, theta, values, cmap='viridis')
    ax.set_yticks(np.arange(0, 190, 45))
    ax.set_xticks(np.arange(0, 370, 45))

    ax.set_xlabel('Azimuthal angle φ (°)')
    ax.set_ylabel('Elevation θ (°)')

    ax.grid(visible=True)
    cbar = fig.colorbar(cs, label='Areal probability', format='')

    # Rotate the colorbar label by 180 degrees
    cbar.ax.yaxis.set_label_coords(2.5,
                                   0.5)  # Adjust the position of the label
    cbar.set_label('Areal probability', rotation=270, labelpad=15)
    return fig
