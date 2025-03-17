from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure

    from gemdat import Jumps


def jumps_vs_distance(
    *,
    jumps: Jumps,
    jump_res: float = 0.1,
    n_parts: int = 1,
) -> matplotlib.figure.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    jumps : Jumps
        Input data
    jump_res : float, optional
        Resolution of the bins in Angstrom
    n_parts : int
        Number of parts for error analysis

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    sites = jumps.sites
    trajectory = jumps.trajectory
    lattice = trajectory.get_lattice()

    pdist = lattice.get_all_distances(sites.frac_coords, sites.frac_coords)

    bin_max = (1 + pdist.max() // jump_res) * jump_res
    n_bins = int(bin_max / jump_res) + 1
    x = np.linspace(0, bin_max, n_bins)

    bin_idx = np.digitize(pdist, bins=x)
    data = []
    for transitions_part in jumps.split(n_parts=n_parts):
        counts = np.zeros_like(x)
        for idx, n in zip(bin_idx.flatten(), transitions_part.matrix().flatten()):
            counts[idx] += n
        for idx in range(n_bins):
            if counts[idx] > 0:
                data.append((x[idx], counts[idx]))

    df = pd.DataFrame(data=data, columns=['Displacement', 'count'])

    grouped = df.groupby(['Displacement'])
    mean = grouped.mean().reset_index().rename(columns={'count': 'mean'})
    std = grouped.std().reset_index().rename(columns={'count': 'std'})
    df = mean.merge(std, how='inner')

    fig, ax = plt.subplots()

    if n_parts == 1:
        ax.bar('Displacement', 'mean', data=df, width=(jump_res * 0.8))
    else:
        ax.bar('Displacement', 'mean', yerr='std', data=df, width=(jump_res * 0.8))

    ax.set(
        title='Jumps vs. Distance',
        xlabel='Distance (Å)',
        ylabel='Number of jumps',
    )

    return fig
