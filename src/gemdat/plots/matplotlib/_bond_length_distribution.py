from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from .._shared import _fit_skewnorm_to_hist, _orientations_to_histogram

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat.orientations import Orientations


def bond_length_distribution(
    *, orientations: Orientations, bins: int = 50
) -> matplotlib.figure.Figure:
    """Plot the bond length probability distribution.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    bins : int, optional
        The number of bins, by default 1000

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    hist_df = _orientations_to_histogram(orientations, bins=bins)

    centers = hist_df['center']
    prob = hist_df['prob']
    bin_width = hist_df['left_edge'].iloc[0] - hist_df['right_edge'].iloc[0]

    x, y = _fit_skewnorm_to_hist(hist_df)

    fig, ax = plt.subplots()

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

    return fig
