from __future__ import annotations

from typing import TYPE_CHECKING

from ._axes import get_axes

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from gemdat import Jumps


def collective_jumps(
    *, jumps: Jumps, ax: matplotlib.axes.Axes | None = None
) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
    """Plot collective jumps per jump-type combination.

    Parameters
    ----------
    jumps : Jumps
        Input data
    ax : matplotlib.axes.Axes | None
        Draw into this axes instead of a new figure; the axes is then returned.

    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes
        Output figure, or `ax` if given.
    """
    fig, ax = get_axes(ax)

    collective = jumps.collective()

    matrix = collective.site_pair_count_matrix()

    im = ax.imshow(matrix)

    labels = collective.site_pair_count_matrix_labels()
    ticks = range(len(labels))

    ax.set_xticks(ticks, labels=labels, rotation=90)
    ax.set_yticks(ticks, labels=labels)

    ax.figure.colorbar(im, ax=ax)

    ax.set(title='Cooperative jumps per jump-type combination')

    return ax if fig is None else fig
