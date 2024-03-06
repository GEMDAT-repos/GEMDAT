from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

from gemdat.rotations import autocorrelation, calculate_spherical_areas


def rectilinear_plot(*,
                     trajectory: np.ndarray,
                     shape: tuple = (90, 360),
                     normalize: bool = True) -> plt.Figure:
    """Plot a rectilinear projection of a spherical function.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory in spherical coordinates
    shape : tuple
        The shape of the spherical sector in which the trajectory is plotted
    normalize : bool, optional
        If True, normalize the histogram by the area of the bins, by default True

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    az = trajectory[:, :, 0].flatten()
    el = trajectory[:, :, 1].flatten()

    hist, xedges, yedges = np.histogram2d(el, az, shape)

    if normalize:
        # Normalize by the area of the bins
        areas = calculate_spherical_areas(shape)
        hist = np.divide(hist, areas)
        #replace values at the poles where normalization breaks - hacky
        hist[0, :] = hist[1, :]
        hist[-1, :] = hist[-2, :]

    values = hist.T
    phi = np.linspace(0, 360, np.ma.size(values, 0))
    theta = np.linspace(0, 180, np.ma.size(values, 1))

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


def _skewed_gaussian(xgrid: np.ndarray, loc: float, scale: float,
                     skew: float) -> np.ndarray:
    """Returnd the probability distribution function of a skewed gaussian
    distribution, using scipy.stats.skewnorm.

    Parameters
    ----------
    xgrid : np.ndarray
        The grid points at which to evaluate the function
    loc : float
        The location parameter which defines the mean of the distribution and
        it is used to shift the distribution along the x-axis
    scale : float
        The scale parameter which defines the standard deviation of the distribution
    skew : float
        The skew parameter which defines the asymmetry of the distribution.
        A positive value indicates a right-skewed distribution, a negative value
        indicates a left-skewed distribution, and a value of 0 indicates a normal distribution.

    Returns
    -------
    np.ndarray
        The value of the skewed Gaussian distribution function at the given grid points
    """
    return skewnorm.pdf(xgrid, a=skew, loc=loc, scale=scale)


def bond_length_distribution(*,
                             direction: np.ndarray,
                             bins: int = 1000) -> plt.Figure:
    """Plot the bond length probability distribution.

    Parameters
    ----------
    direction : np.ndarray
        The direction of the bonds in polar coordinates
    bins : int, optional
        The number of bins, by default 1000

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    fig, ax = plt.subplots()

    bond_lengths = direction[:, :, 2].flatten()

    # Plot the normalized histogram
    hist, edges = np.histogram(bond_lengths, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Fit a skewed Gaussian distribution to the data
    params, covariance = curve_fit(_skewed_gaussian,
                                   bin_centers,
                                   hist,
                                   p0=[1.5, 1, 1.5])

    # Plot the histogram
    ax.hist(bond_lengths,
            bins=bins,
            density=True,
            color='blue',
            alpha=0.7,
            label='Data')

    # Plot the fitted skewed Gaussian distribution
    x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
    ax.plot(x_fit,
            _skewed_gaussian(x_fit, *params),
            'r-',
            label='Skewed Gaussian Fit')

    ax.set_xlabel(r'Bond length $[\AA]$')
    ax.set_ylabel(r'Probability density $[\AA^{-1}]$')
    ax.set_title('Bond Length Probability Distribution')
    ax.legend()
    ax.grid(True)

    return fig


def unit_vector_autocorrelation(*, trajectory: np.ndarray,
                                time_units: float) -> plt.Figure:
    """Plot the autocorrelation function of the unit vectors series.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory in direct cartesian coordinates. It is expected
        to have shape (n_times, n_particles, n_coordinates)
    time_units : float
        The time step of the simulation in seconds, the default unit of pymatgen.trajectory.time_unit

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

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
