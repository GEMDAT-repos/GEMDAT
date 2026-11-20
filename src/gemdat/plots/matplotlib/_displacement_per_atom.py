from __future__ import annotations

from typing import TYPE_CHECKING

from ._axes import get_axes

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from gemdat.trajectory import Trajectory


def displacement_per_atom(
    *, trajectory: Trajectory, ax: matplotlib.axes.Axes | None = None
) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
    """Plot displacement per atom.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom
    ax : matplotlib.axes.Axes | None
        Draw into this axes instead of a new figure; the axes is then returned.

    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes
        Output figure, or `ax` if given.
    """
    fig, ax = get_axes(ax)

    for distances in trajectory.distances_from_base_position():
        ax.plot(distances, lw=0.3)

    ax.set(title='Displacement per site', xlabel='Time step', ylabel='Displacement (Å)')

    return ax if fig is None else fig
