from __future__ import annotations

from typing import TYPE_CHECKING

from .._shared import _jumps_vs_distance
from ._axes import get_axes

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from gemdat.jumps import Jumps


def jumps_vs_distance(
    *,
    jumps: Jumps,
    jump_res: float = 0.1,
    n_parts: int = 1,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
    """Plot jumps vs distance histogram.

    Parameters
    ----------
    jumps : Jumps
        Input data
    jump_res : float, optional
        Resolution of the bins in Angstrom
    n_parts : int
        Number of parts for error analysis
    ax : matplotlib.axes.Axes | None
        Draw into this axes instead of a new figure; the axes is then returned.

    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes
        Output figure, or `ax` if given.
    """
    df = _jumps_vs_distance(jumps=jumps, resolution=jump_res, n_parts=n_parts)

    fig, ax = get_axes(ax)

    if n_parts == 1:
        ax.bar('Displacement', 'mean', data=df, width=(jump_res * 0.8))
    else:
        ax.bar('Displacement', 'mean', yerr='std', data=df, width=(jump_res * 0.8))

    ax.set(
        title='Jumps vs. Distance',
        xlabel='Distance (Å)',
        ylabel='Number of jumps',
    )

    return ax if fig is None else fig
