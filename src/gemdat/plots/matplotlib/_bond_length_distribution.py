from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

from gemdat.orientations import Orientations


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

    ax.set_xlabel('Bond length (Å)')
    ax.set_ylabel(r'Probability density (Å$^{-1}$)')
    ax.set_title('Bond Length Probability Distribution')
    ax.legend()
    ax.grid(True)

    return fig
