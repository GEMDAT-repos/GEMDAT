from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat.rdf import RDFData


def radial_distribution(rdfs: Iterable[RDFData]) -> matplotlib.figure.Figure:
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
        ax.plot(rdf.x, rdf.y, label=rdf.label)

    states = ', '.join({rdf.state for rdf in rdfs if rdf.state})
    state_suffix = f' ({states})' if states else ''

    ax.legend()
    ax.set(
        title=f'Radial distribution function{state_suffix}',
        xlabel='Distance (Å)',
        ylabel='Counts',
    )

    return fig
