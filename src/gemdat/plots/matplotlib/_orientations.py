from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

from gemdat.orientations import (
    Orientations, )


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

    fig, ax = plt.subplots(subplot_kw=dict(projection='rectilinear'))
    cs = ax.contourf(phi, theta, values, cmap='viridis')
    ax.set_yticks(np.arange(0, 190, 45))
    ax.set_xticks(np.arange(0, 370, 45))

    ax.set_xlabel(r'azimuthal angle φ $[\degree$]')
    ax.set_ylabel(r'elevation θ $[\degree$]')

    ax.grid(visible=True)
    cbar = fig.colorbar(cs, label='areal probability', format='')

    if add_peaks:
        peaks = ov.orientational_peaks(**kwargs)
        xp = [x for x, _ in peaks]
        yp = [y for _, y in peaks]
        ax.plot(xp, yp, 'ro', markersize=5)

    # Rotate the colorbar label by 180 degrees
    cbar.ax.yaxis.set_label_coords(2.5,
                                   0.5)  # Adjust the position of the label
    cbar.set_label('areal probability', rotation=270, labelpad=15)
    return fig


def bond_length_distribution(*,
                             orientations: Orientations,
                             bins: int = 1000) -> plt.Figure:
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
    *_, bond_lengths = orientations.vectors_spherical.T
    bond_lengths = bond_lengths.flatten()

    fig, ax = plt.subplots()

    # Plot the normalized histogram
    hist, edges = np.histogram(bond_lengths, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Fit a skewed Gaussian distribution to the orientations
    params, covariance = curve_fit(skewnorm.pdf,
                                   bin_centers,
                                   hist,
                                   p0=[1.5, 1, 1.5])

    # Create a new function using the fitted parameters
    def _skewnorm_fit(x):
        return skewnorm.pdf(x, *params)

    # Plot the histogram
    ax.hist(bond_lengths,
            bins=bins,
            density=True,
            color='blue',
            alpha=0.7,
            label='Data')

    # Plot the fitted skewed Gaussian distribution
    x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
    ax.plot(x_fit, _skewnorm_fit(x_fit), 'r-', label='Skewed Gaussian Fit')

    ax.set_xlabel(r'Bond length $[\AA]$')
    ax.set_ylabel(r'Probability density $[\AA^{-1}]$')
    ax.set_title('Bond Length Probability Distribution')
    ax.legend()
    ax.grid(True)

    return fig


def autocorrelation(
    *,
    orientations: Orientations,
) -> plt.Figure:
    """Plot the autocorrelation function of the unit vectors series.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    ac = orientations.autocorrelation()
    ac_std = ac.std(axis=0)
    ac_mean = ac.mean(axis=0)

    # Since we want to plot in picosecond, we convert the time units
    time_ps = orientations._time_step * 1e12
    tgrid = np.arange(ac_mean.shape[0]) * time_ps

    # and now we can plot the autocorrelation function
    fig, ax = plt.subplots()

    ax.plot(tgrid, ac_mean, label='FFT-Autocorrelation')
    ax.fill_between(tgrid, ac_mean - ac_std, ac_mean + ac_std, alpha=0.2)
    ax.set_xlabel('Time lag [ps]')
    ax.set_ylabel('Autocorrelation')

    return fig
