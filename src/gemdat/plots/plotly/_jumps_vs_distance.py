from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go

from .._shared import _jumps_vs_distance

if TYPE_CHECKING:
    from gemdat import Jumps


def jumps_vs_distance(
    *,
    jumps: Jumps,
    jump_res: float = 0.1,
    n_parts: int = 1,
) -> go.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    jumps : Jumps
        Input jumps data
    jump_res : float, optional
        Resolution of the bins in Angstrom
    n_parts : int
        Number of parts for error analysis

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    df = _jumps_vs_distance(jumps=jumps, resolution=jump_res, n_parts=n_parts)

    if n_parts == 1:
        fig = px.bar(df, x='Displacement', y='mean', barmode='stack')
    else:
        fig = px.bar(df, x='Displacement', y='mean', error_y='std', barmode='stack')

    fig.update_layout(
        title='Jumps vs. Distance',
        xaxis_title='Distance (Å)',
        yaxis_title='Number of jumps',
    )

    return fig
