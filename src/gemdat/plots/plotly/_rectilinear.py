from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

if TYPE_CHECKING:
    from gemdat.orientations import Orientations


def rectilinear(
    *,
    orientations: Orientations,
    shape: tuple[int, int] = (90, 360),
    normalize_histo: bool = True,
) -> go.Figure:
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
    fig : plotly.graph_objects.Figure
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

    fig = go.Figure(
        data=go.Contour(
            x=phi,
            y=theta,
            z=hist,
            colorbar={
                'title': 'Areal probability',
                'title_side': 'right',
            },
        )
    )

    fig.update_layout(
        title='Rectilinear plot',
        xaxis_title='Azimuthal angle φ (°)',
        yaxis_title='Elevation θ (°)',
    )

    return fig
