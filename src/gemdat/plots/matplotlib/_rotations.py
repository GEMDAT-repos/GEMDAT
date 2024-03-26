from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

from gemdat.rotations import Orientations, calculate_spherical_areas, mean_squared_angular_displacement
from gemdat.utils import cartesian_to_spherical


def rectilinear_plot(*,
                     orientations: Orientations,
                     shape: tuple[int, int] = (90, 360),
                     symmetrize: bool = True,
                     normalize: bool = True) -> plt.Figure:
    """Plot a rectilinear projection of a spherical function.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    symmetrize : bool, optional
        If True, use the symmetrized trajectory
    shape : tuple
        The shape of the spherical sector in which the trajectory is plotted
    normalize : bool, optional
        If True, normalize the histogram by the area of the bins, by default True

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    if symmetrize:
        trajectory = orientations.get_symmetric_trajectory()
    else:
        trajectory = orientations.get_conventional_coordinates()
    # Convert the trajectory to spherical coordinates
    trajectory = cartesian_to_spherical(trajectory, degrees=True)

    az = trajectory[:, :, 0].flatten()
    el = trajectory[:, :, 1].flatten()

    hist, xedges, yedges = np.histogram2d(el, az, shape)

    if normalize:
        # Normalize by the area of the bins
        areas = calculate_spherical_areas(shape)
        hist = np.divide(hist, areas)
        # Drop the bins at the poles where normalization is not possible
        hist = hist[1:-1, :]

    values = hist.T
    axis_phi, axis_theta = values.shape

    phi = np.linspace(0, 360, axis_phi)
    theta = np.linspace(0, 180, axis_theta)

    theta, phi = np.meshgrid(theta, phi)

    fig, ax = plt.subplots(subplot_kw=dict(projection='rectilinear'))
    cs = ax.contourf(phi, theta, values, cmap='viridis')
    ax.set_yticks(np.arange(0, 190, 45))
    ax.set_xticks(np.arange(0, 370, 45))

    ax.set_xlabel(r'azimuthal angle φ $[\degree$]')
    ax.set_ylabel(r'elevation θ $[\degree$]')

    ax.grid(visible=True)
    cbar = fig.colorbar(cs, label='areal probability', format='')

    # Rotate the colorbar label by 180 degrees
    cbar.ax.yaxis.set_label_coords(2.5,
                                   0.5)  # Adjust the position of the label
    cbar.set_label('areal probability', rotation=270, labelpad=15)
    return fig


def bond_length_distribution(*,
                             orientations: Orientations,
                             symmetrize: bool = True,
                             bins: int = 1000) -> plt.Figure:
    """Plot the bond length probability distribution.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    symmetrize : bool, optional
        If True, use the symmetrized trajectory
    bins : int, optional
        The number of bins, by default 1000

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    if symmetrize:
        trajectory = orientations.get_symmetric_trajectory()
    else:
        trajectory = orientations.get_conventional_coordinates()
    # Convert the trajectory to spherical coordinates
    trajectory = cartesian_to_spherical(trajectory, degrees=True)

    fig, ax = plt.subplots()

    bond_lengths = trajectory[:, :, 2].flatten()

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


def unit_vector_autocorrelation(
    *,
    orientations: Orientations,
    n_tgrid: int = -1,
) -> plt.Figure:
    """Plot the autocorrelation function of the unit vectors series.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    n_tgrid : int, optional
        Number of time points to use for the autocorrelation function.
        If -1 (default), all time points are used via the FFT algorithm.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    # The trajectory is expected to have shape (n_times, n_particles, n_coordinates)
    trajectory = orientations.get_unit_vectors_trajectory()

    ac, tgrid = mean_squared_angular_displacement(trajectory, n_tgrid=n_tgrid)
    ac_mean = ac.mean(axis=0)
    ac_std = ac.std(axis=0)

    # Since we want to plot in picosecond, we convert the time units
    time_ps = orientations._time_step * 1e12
    tgrid = tgrid * time_ps

    # and now we can plot the autocorrelation function
    fig, ax = plt.subplots()

    ax.plot(tgrid, ac_mean, label='FFT-Autocorrelation')
    ax.fill_between(tgrid, ac_mean - ac_std, ac_mean + ac_std, alpha=0.2)
    ax.set_xlabel('Time lag [ps]')
    ax.set_ylabel('Autocorrelation')

    return fig
