from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from gemdat import Jumps


def jumps_3d(*, jumps: Jumps) -> go.Figure:
    """Plot jumps in 3D.

    Parameters
    ----------
    jumps : Jumps
        Input data

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    from ._plot3d import plot_3d

    return plot_3d(jumps=jumps, structure=jumps.sites)
