from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Element, Species

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat.trajectory import Trajectory


def msd_per_element(
    *,
    trajectory: Trajectory,
    show_traces: bool = True,
    show_shaded: bool = True,
) -> matplotlib.figure.Figure:
    """Plot mean squared displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory
    show_traces : bool
        If True, show traces of individual trajectories for each element
    show_shaded : bool
        If True, show standard deviation as shaded area

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    species = list(set(trajectory.species))

    fig, ax = plt.subplots()

    time_ps = trajectory.time_step_ps
    t_values = np.arange(len(trajectory)) * time_ps

    for sp in species:
        assert isinstance(sp, (Species, Element)), f'got {type(sp)}'

        traj = trajectory.filter(sp.symbol)
        msd = traj.mean_squared_displacement()

        msd_mean = np.mean(msd, axis=0)
        msd_std = np.std(msd, axis=0)

        ax.plot(t_values, msd_mean, lw=0.5, label=f'{sp.symbol} mean')

        last_color = ax.lines[-1].get_color()

        if show_traces:
            for i, y_values in enumerate(msd):
                label = f'{sp.symbol} trajectories' if (i == 0) else None
                ax.plot(t_values, y_values, lw=0.1, c=last_color, label=label)

        if show_shaded:
            ax.fill_between(
                t_values,
                msd_mean - msd_std,
                msd_mean + msd_std,
                color=last_color,
                alpha=0.2,
                label=f'{sp.symbol} std',
            )

    ax.legend()
    ax.set(
        title='Mean squared displacement per element',
        xlabel='Time lag (ps)',
        ylabel='MSD (Å$^2$)',
    )

    return fig
