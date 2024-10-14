from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from gemdat.plots._shared import _mean_displacements_per_element

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat.trajectory import Trajectory


def displacement_per_element(*, trajectory: Trajectory) -> matplotlib.figure.Figure:
    """Plot displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    displacements = _mean_displacements_per_element(trajectory)

    fig, ax = plt.subplots()

    for symbol, (mean, _) in displacements.items():
        ax.plot(mean, lw=0.3, label=symbol)

    ax.legend()
    ax.set(title='Displacement per element', xlabel='Time step', ylabel='Displacement (Å)')

    return fig
