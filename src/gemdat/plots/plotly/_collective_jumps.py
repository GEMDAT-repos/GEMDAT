from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:
    from gemdat import Jumps


def collective_jumps(*, jumps: Jumps) -> go.Figure:
    """Plot collective jumps per jump-type combination.

    Parameters
    ----------
    jumps : Jumps
        Input data

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """

    matrix = jumps.collective().site_pair_count_matrix()
    labels = jumps.collective().site_pair_count_matrix_labels()

    fig = px.imshow(matrix)

    ticks = list(range(len(labels)))

    fig.update_layout(xaxis={
        'tickmode': 'array',
        'tickvals': ticks,
        'ticktext': labels
    },
                      yaxis={
                          'tickmode': 'array',
                          'tickvals': ticks,
                          'ticktext': labels
                      },
                      title='Cooperative jumps per jump-type combination')

    return fig


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
