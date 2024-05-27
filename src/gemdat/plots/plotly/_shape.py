from __future__ import annotations

from typing import TYPE_CHECKING, Collection, Sequence

import plotly.graph_objects as go

if TYPE_CHECKING:
    from pymatgem.core import PeriodicSite

    from gemdat.shape import ShapeData


def shape(
    shape: ShapeData,
    bins: int | Sequence[float] = 50,
    sites: Collection[PeriodicSite] | None = None,
) -> go.Figure:
    """Plot site cluster shapes.

    Parameters
    ----------
    shape : ShapeData
        Shape data to plot
    bins : int | Sequence[float]
        Number of bins or sequence of bin edges.
        See [hist()][matplotlib.pyplot.hist] for more info.
    sites : Collection[PeriodicSite] | None
        Plot these sites on the shape density

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    raise NotImplementedError
