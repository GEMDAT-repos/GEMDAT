from __future__ import annotations

import plotly.graph_objects as go

from gemdat.orientations import Orientations


def rectilinear(*,
                orientations: Orientations,
                shape: tuple[int, int] = (90, 360),
                normalize_histo: bool = True) -> go.Figure:
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
    raise NotImplementedError
