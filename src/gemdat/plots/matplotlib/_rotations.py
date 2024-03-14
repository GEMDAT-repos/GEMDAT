from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

from gemdat.rotations import Orientations, autocorrelation, calculate_spherical_areas
from gemdat.utils import cartesian_to_spherical


def rectilinear_plot(*,
                     data: Orientations,
                     shape: tuple[int, int] = (90, 360),
                     symmetrize: bool = True,
                     normalize: bool = True) -> plt.Figure:
    """Plot a rectilinear projection of a spherical function.

    Parameters
    ----------
    data : Orientations
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
        trajectory = data.get_symmetric_traj()
    else:
        trajectory = data.get_conventional_coordinates()
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
                             data: Orientations,
                             symmetrize: bool = True,
                             bins: int = 1000) -> plt.Figure:
    """Plot the bond length probability distribution.

    Parameters
    ----------
    data : Orientations
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
        trajectory = data.get_symmetric_traj()
    else:
        trajectory = data.get_conventional_coordinates()
    # Convert the trajectory to spherical coordinates
    trajectory = cartesian_to_spherical(trajectory, degrees=True)

    fig, ax = plt.subplots()

    bond_lengths = trajectory[:, :, 2].flatten()

    # Plot the normalized histogram
    hist, edges = np.histogram(bond_lengths, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Fit a skewed Gaussian distribution to the data
    params, covariance = curve_fit(
        lambda x, a, loc, scale: skewnorm.pdf(x, a, loc, scale),
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


def unit_vector_autocorrelation(*, data: Orientations,
                                time_units: float) -> plt.Figure:
    """Plot the autocorrelation function of the unit vectors series.

    Parameters
    ----------
    data : Orientations
        The unit vector trajectories
    time_units : float
        The time step of the simulation in seconds, the default unit of pymatgen.trajectory.time_unit

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    # The trajectory is expected to have shape (n_times, n_particles, n_coordinates)
    trajectory = data.get_unit_vectors_traj()

    ac, std_ac = autocorrelation(trajectory)

    # Since we want to plot in picosecond, we convert the time units
    time_ps = time_units * 1e12
    t_values = np.arange(len(ac)) * time_ps

    # and now we can plot the autocorrelation function
    fig, ax = plt.subplots()

    ax.plot(t_values, ac, label='FFT-Autocorrelation')
    ax.fill_between(t_values, ac - std_ac, ac + std_ac, alpha=0.2)
    ax.set_xlabel('Time lag [ps]')
    ax.set_ylabel('Autocorrelation')

    return fig
