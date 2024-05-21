from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from gemdat import Jumps


def jumps_vs_distance(
    *,
    jumps: Jumps,
    jump_res: float = 0.1,
) -> plt.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    jumps : Jumps
        Input data
    jump_res : float, optional
        Resolution of the bins in Angstrom

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
    counts = np.zeros_like(x)

    bin_idx = np.digitize(pdist, bins=x)
    for idx, n in zip(bin_idx.flatten(), jumps.matrix().flatten()):
        counts[idx] += n

    fig, ax = plt.subplots()

    ax.bar(x, counts, width=(jump_res * 0.8))

    ax.set(title='Jumps vs. Distance',
           xlabel='Distance (Å)',
           ylabel='Number of jumps')

    return fig
