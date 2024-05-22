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
    raise NotImplementedError
