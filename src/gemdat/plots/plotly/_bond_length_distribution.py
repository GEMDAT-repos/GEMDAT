from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go

from .._shared import _fit_skewnorm_to_hist, _orientations_to_histogram

if TYPE_CHECKING:
    from gemdat.orientations import Orientations


def bond_length_distribution(*, orientations: Orientations, bins: int = 50) -> go.Figure:
    """Plot the bond length probability distribution.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    bins : int, optional
        The number of bins

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    hist_df = _orientations_to_histogram(orientations, bins=bins)
    x, y = _fit_skewnorm_to_hist(hist_df, steps=100)

    fig = px.bar(
        hist_df,
        x='center',
        y='prob',
    )

    fig.add_trace(
        go.Scatter(x=x, y=y, name='Skewed Gaussian Fit', mode='lines', line={'width': 3})
    )

    fig.update_layout(
        title='Bond length probability distribution',
        xaxis_title='Bond length (Å)',
        yaxis_title='Probability density (Å<sup>-1</sup>)',
    )

    return fig
