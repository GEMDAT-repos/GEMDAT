from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from ._axes import get_axes

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from gemdat.rdf import RDFData


def radial_distribution(
    rdfs: Iterable[RDFData], *, ax: matplotlib.axes.Axes | None = None
) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
    """Plot radial distribution function.

    Parameters
    ----------
    rdfs : Iterable[RDFData]
        List of RDF data to plot
    ax : matplotlib.axes.Axes | None
        Draw into this axes instead of a new figure; the axes is then returned.

    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes
        Output figure, or `ax` if given.
    """
    fig, ax = get_axes(ax)

    for rdf in rdfs:
        ax.plot(rdf.x, rdf.y, label=rdf.label)

    states = ', '.join({rdf.state for rdf in rdfs if rdf.state})
    state_suffix = f' ({states})' if states else ''

    ax.legend()
    ax.set(
        title=f'Radial distribution function{state_suffix}',
        xlabel='Distance (Å)',
        ylabel='g(r)',
    )

    return ax if fig is None else fig
