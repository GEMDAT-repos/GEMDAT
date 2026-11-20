from __future__ import annotations

from typing import TYPE_CHECKING

from .._shared import _fit_skewnorm_to_hist, _orientations_to_histogram
from ._axes import get_axes

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from gemdat.orientations import Orientations


def bond_length_distribution(
    *,
    orientations: Orientations,
    bins: int = 50,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
    """Plot the bond length probability distribution.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    bins : int, optional
        The number of bins, by default 1000
    ax : matplotlib.axes.Axes | None
        Draw into this axes instead of a new figure; the axes is then returned.

    Returns
    -------
    fig : matplotlib.figure.Figure | matplotlib.axes.Axes
        Output figure, or `ax` if given.
    """
    hist_df = _orientations_to_histogram(orientations, bins=bins)

    centers = hist_df['center']
    prob = hist_df['prob']
    bin_width = hist_df['left_edge'].iloc[0] - hist_df['right_edge'].iloc[0]

    x, y = _fit_skewnorm_to_hist(hist_df)

    fig, ax = get_axes(ax)

    ax.plot(x, y, 'r-', lw=2, label='Skewed Gaussian Fit')

    ax.bar(
        x=centers,
        height=prob,
        width=bin_width * 0.8,
        edgecolor='none',
        alpha=0.7,
        label='Probability',
    )

    ax.set_xlabel('Bond length (Å)')
    ax.set_ylabel(r'Probability density (Å$^{-1}$)')
    ax.set_title('Bond length probability distribution')
    ax.legend()

    return ax if fig is None else fig
