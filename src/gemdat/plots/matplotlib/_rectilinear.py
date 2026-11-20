from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._axes import get_axes

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from gemdat.orientations import Orientations


def rectilinear(
    *,
    orientations: Orientations,
    shape: tuple[int, int] = (90, 360),
    normalize_histo: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
    """Plot a rectilinear projection of a spherical function.

    This function uses the transformed trajectory.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    shape : tuple
        The shape of the spherical sector in which the trajectory is plotted
    normalize_histo : bool, optional
        If True, normalize the histogram by the area of the bins, by default True
    ax : matplotlib.axes.Axes | None
        Draw into this axes instead of a new figure; the axes is then returned.

    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes
        Output figure, or `ax` if given.
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

    fig, ax = get_axes(ax)

    cs = ax.contourf(x, y, hist)

    ax.set_xticks(np.linspace(0, 360, 9))
    ax.set_yticks(np.linspace(0, 180, 5))

    ax.set_xlabel('Azimuthal angle φ (°)')
    ax.set_ylabel('Elevation θ (°)')

    ax.figure.colorbar(cs, ax=ax, label='Areal probability', format='')

    return ax if fig is None else fig
