from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import plotly.graph_objects as go

if TYPE_CHECKING:
    from gemdat.rdf import RDFData


def radial_distribution(rdfs: Iterable[RDFData]) -> go.Figure:
    """Plot radial distribution function.

    Parameters
    ----------
    rdfs : Iterable[RDFData]
        List of RDF data to plot

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    fig = go.Figure()

    for rdf in rdfs:
        fig.add_trace(
            go.Scatter(
                x=rdf.x,
                y=rdf.y,
                name=rdf.label,
                mode='lines',
                # line={'width': 0.25}
            )
        )

    states = ', '.join({rdf.state for rdf in rdfs if rdf.state})
    state_suffix = f' ({states})' if states else ''

    fig.update_layout(
        title=f'Radial distribution function{state_suffix}',
        xaxis_title='Distance (Å)',
        yaxis_title='Counts',
    )

    return fig
