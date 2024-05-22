from __future__ import annotations
import plotly.graph_objects as go
from gemdat.orientations import Orientations


def autocorrelation(
    *,
    orientations: Orientations,
    show_traces: bool = True,
    show_shaded: bool = True,
) -> go.Figure:
    """Plot the autocorrelation function of the unit vectors series.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    show_traces : bool
        If True, show traces of individual trajectories
    show_shaded : bool
        If True, show standard deviation as shaded area

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    raise NotImplementedError
