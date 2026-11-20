from __future__ import annotations

from typing import TYPE_CHECKING

from gemdat.plots._shared import _mean_displacements_per_element

from ._axes import get_axes

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from gemdat.trajectory import Trajectory


def displacement_per_element(
    *, trajectory: Trajectory, ax: matplotlib.axes.Axes | None = None
) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
    """Plot displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    ax : matplotlib.axes.Axes | None
        Draw into this axes instead of a new figure; the axes is then returned.

    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes
        Output figure, or `ax` if given.
    """
    displacements = _mean_displacements_per_element(trajectory)

    fig, ax = get_axes(ax)

    for symbol, (mean, _) in displacements.items():
        ax.plot(mean, lw=0.3, label=symbol)

    ax.legend()
    ax.set(title='Displacement per element', xlabel='Time step', ylabel='Displacement (Å)')

    return ax if fig is None else fig
