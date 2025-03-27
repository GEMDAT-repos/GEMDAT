from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat.orientations import Orientations


def polar(
    *,
    orientations: Orientations,
    shape: tuple[int, int] = (90, 360),
    normalize_histo: bool = True,
) -> matplotlib.figure.Figure:
    """Plot a polar projection of a spherical function. This function uses the
    transformed trajectory.

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

    phi = np.radians(np.linspace(0, 360, axis_phi))
    theta = np.linspace(0, 180, axis_theta)

    theta, phi = np.meshgrid(theta, phi)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))

    cs1 = ax1.contourf(phi, theta, hist.T)
    ax1.set_title('θ < 90°')
    ax1.set_rmax(90)
    ax1.set_yticklabels([])

    ax2.contourf(phi, 180 - theta, hist.T)
    ax2.set_title('θ > 90°')
    ax2.set_rmax(90)
    ax2.set_yticklabels([])

    fig.colorbar(cs1, ax=[ax1, ax2], orientation='horizontal', label='Areal Probability')

    plt.subplots_adjust(wspace=0.5, bottom=0.35)  # Increase horizontal spacing

    return fig
