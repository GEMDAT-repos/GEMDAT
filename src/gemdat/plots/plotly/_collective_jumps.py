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
    collective = jumps.collective()
    matrix = collective.site_pair_count_matrix()

    fig = px.imshow(matrix)

    labels = collective.site_pair_count_matrix_labels()

    ticks = list(range(len(labels)))

    fig.update_layout(
        xaxis={'tickmode': 'array', 'tickvals': ticks, 'ticktext': labels},
        yaxis={'tickmode': 'array', 'tickvals': ticks, 'ticktext': labels},
        title='Cooperative jumps per jump-type combination',
    )

    return fig
