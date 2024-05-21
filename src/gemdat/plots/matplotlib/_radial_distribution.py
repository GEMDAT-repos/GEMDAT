from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from gemdat.rdf import RDFData


def radial_distribution(rdfs: Iterable[RDFData]) -> plt.Figure:
    """Plot radial distribution function.

    Parameters
    ----------
    rdfs : Iterable[RDFData]
        List of RDF data to plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()

    for rdf in rdfs:
        ax.plot(rdf.x, rdf.y, label=rdf.symbol)

    states = ', '.join({rdf.state for rdf in rdfs})

    ax.legend()
    ax.set(title=f'Radial distribution function ({states})',
           xlabel='Distance (Ang)',
           ylabel='Counts')

    return fig
