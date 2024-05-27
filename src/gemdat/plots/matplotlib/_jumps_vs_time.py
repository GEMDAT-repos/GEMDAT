from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from gemdat import Jumps


def jumps_vs_time(*, jumps: Jumps, binsize: int = 500) -> plt.Figure:
    """Plot jumps vs. time histogram.

    Parameters
    ----------
    jumps : Jumps
        Input data
    binsize : int, optional
        Width of each bin in number of time steps

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    trajectory = jumps.trajectory

    n_steps = len(trajectory)
    bins = np.arange(0, n_steps + binsize, binsize)

    fig, ax = plt.subplots()

    ax.hist(jumps.data['stop time'], bins=bins, width=0.8 * binsize)

    ax.set(title='Jumps vs. time', xlabel='Time (steps)', ylabel='Number of jumps')

    return fig
