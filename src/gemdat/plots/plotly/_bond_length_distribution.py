from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px

from gemdat.orientations import Orientations
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skewnorm


def bond_length_distribution(*,
                             orientations: Orientations,
                             bins: int = 1000) -> go.Figure:
    """Plot the bond length probability distribution.

    Parameters
    ----------
    orientations : Orientations
        The unit vector trajectories
    bins : int, optional
        The number of bins, by default 1000

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    *_, bond_lengths = orientations.vectors_spherical.T
    bond_lengths = bond_lengths.flatten()

    hist, edges = np.histogram(bond_lengths, bins=bins, density=True)
    bin_centers = bin_centers = (edges[:-1] + edges[1:]) / 2

    params, covariance = curve_fit(skewnorm.pdf,
                                   bin_centers,
                                   hist,
                                   p0=[1.5, 1, 1.5])

    x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
    y_fit = skewnorm.pdf(x_fit, *params)

    df = pd.DataFrame({'prob': hist, 'x': bin_centers})

    fig = px.bar(
        df,
        x='x',
        y='prob',
    )

    fig.add_trace(
        go.Scatter(x=x_fit,
                   y=y_fit,
                   name='Skewed Gaussian Fit',
                   mode='lines',
                   line={'width': 3}))

    fig.update_layout(title='Bond Length Probability Distribution',
                      xaxis_title='Bond length (Å)',
                      yaxis_title='Probability density (Å<sup>-1</sup>)')

    return fig
