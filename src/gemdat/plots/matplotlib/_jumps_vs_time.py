from __future__ import annotations

from typing import TYPE_CHECKING

from ._axes import get_axes

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from gemdat import Jumps


def jumps_vs_time(
    *, jumps: Jumps, binsize: int = 500, ax: matplotlib.axes.Axes | None = None
) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
    """Plot jumps vs time histogram.

    Parameters
    ----------
    jumps : Jumps
        Input data
    binsize : int, optional
        Width of each bin in number of time steps
    ax : matplotlib.axes.Axes | None
        Draw into this axes instead of a new figure; the axes is then returned.

    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes
        Output figure, or `ax` if given.
    """
    trajectory = jumps.trajectory

    n_steps = len(trajectory)
    bins = range(0, n_steps + binsize, binsize)

    fig, ax = get_axes(ax)

    ax.hist(jumps.data['stop time'], bins=bins, width=0.8 * binsize)

    ax.set(title='Jumps vs. time', xlabel='Time (steps)', ylabel='Number of jumps')

    return ax if fig is None else fig
