from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat.orientations import Orientations


def rectilinear(
    *,
    orientations: Orientations,
    shape: tuple[int, int] = (90, 360),
    normalize_histo: bool = True,
) -> matplotlib.figure.Figure:
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
    from gemdat.orientations import calculate_spherical_areas

    az, el, _ = orientations.vectors_spherical.T
    az = az.flatten()
    el = el.flatten()

    hist, *_ = np.histogram2d(el, az, shape)

    if normalize_histo:
        areas = calculate_spherical_areas(shape)
        hist = hist / areas
        # Drop the bins at the poles where normalization is not possible
        hist = hist[1:-1, :]

    axis_theta, axis_phi = hist.shape

    phi = np.linspace(0, 360, axis_phi)
    theta = np.linspace(0, 180, axis_theta)

    x, y = np.meshgrid(phi, theta)

    fig, ax = plt.subplots()

    cs = ax.contourf(x, y, hist)

    ax.set_xticks(np.linspace(0, 360, 9))
    ax.set_yticks(np.linspace(0, 180, 5))

    ax.set_xlabel('Azimuthal angle φ (°)')
    ax.set_ylabel('Elevation θ (°)')

    fig.colorbar(cs, label='Areal probability', format='')

    return fig
