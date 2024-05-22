from __future__ import annotations

import plotly.graph_objects as go

from gemdat.path import Pathway


def path_on_grid(path: Pathway) -> go.Figure:
    """Plot the 3d coordinates of the points that define a path.

    Parameters
    ----------
    path : Pathway
        Pathway to plot

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    raise NotImplementedError
