from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gemdat.orientations import Orientations


def rectilinear(
    *,
    orientations: Orientations,
    shape: tuple[int, int] = (90, 360),
    normalize_histo: bool = True,
    add_peaks: bool = False,
    **kwargs,
) -> plt.Figure:
    """Plot a rectilinear projection of a spherical function. This function
    uses the transformed trajectory.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    shape : tuple
        The shape of the spherical sector in which the trajectory is plotted
    normalize_histo : bool, optional
        If True, normalize the histogram by the area of the bins, by default True
    add_peaks : bool, optional
        If True, plot the peaks of the histogram
    **kwargs : dict
        Additional keyword arguments for the `orientational_peaks` method

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    ov = orientations.to_volume(shape=shape, normalize_area=normalize_histo)
    theta, phi, values = ov.data.T

    fig, ax = plt.subplots()
    cs = ax.contourf(phi, theta, values)

    ax.set_xticks(np.linspace(0, 360, 9))
    ax.set_yticks(np.linspace(0, 180, 5))

    ax.set_xlabel('Azimuthal angle φ (°)')
    ax.set_ylabel('Elevation θ (°)')

    fig.colorbar(cs, label='Areal probability', format='')

    if add_peaks:
        peaks = ov.orientational_peaks(**kwargs)
        xp, yp = zip(*peaks)
        ax.plot(xp, yp, 'ro', markersize=5)

    return fig
