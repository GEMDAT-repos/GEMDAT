from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:

    from gemdat import Jumps


def jumps_vs_distance(*,
                      jumps: Jumps,
                      jump_res: float = 0.1,
                      n_parts: int = 1) -> go.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    sites : SitesData
        Input sites data
    jump_res : float, optional
        Resolution of the bins in Angstrom
    n_parts : int
        Number of parts for error analysis

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    sites = jumps.sites
    structure = sites.structure
    trajectory = sites.trajectory
    lattice = trajectory.get_lattice()

    pdist = lattice.get_all_distances(structure.frac_coords,
                                      structure.frac_coords)

    bin_max = (1 + pdist.max() // jump_res) * jump_res
    n_bins = int(bin_max / jump_res) + 1
    x = np.linspace(0, bin_max, n_bins)

    bin_idx = np.digitize(pdist, bins=x)
    data = []
    for transitions_part in jumps.split(n_parts=n_parts):
        counts = np.zeros_like(x)
        for idx, n in zip(bin_idx.flatten(),
                          transitions_part.matrix().flatten()):
            counts[idx] += n
        for idx in range(n_bins):
            if counts[idx] > 0:
                data.append((x[idx], counts[idx]))

    df = pd.DataFrame(data=data, columns=['Displacement', 'count'])

    grouped = df.groupby(['Displacement'])
    mean = grouped.mean().reset_index().rename(columns={'count': 'mean'})
    std = grouped.std().reset_index().rename(columns={'count': 'std'})
    df = mean.merge(std, how='inner')

    df['specie'] = sites.floating_specie

    if n_parts == 1:
        fig = px.bar(df,
                     x='Displacement',
                     y='mean',
                     color='specie',
                     barmode='stack')
    else:
        fig = px.bar(df,
                     x='Displacement',
                     y='mean',
                     color='specie',
                     error_y='std',
                     barmode='stack')

    fig.update_layout(title='Jumps vs. Distance',
                      xaxis_title='Distance (Angstrom)',
                      yaxis_title='Number of jumps')

    return fig


def jumps_vs_time(*,
                  jumps: Jumps,
                  bins: int = 8,
                  n_parts: int = 1) -> go.Figure:
    """Plot jumps vs. distance histogram.

    Parameters
    ----------
    sites : SitesData
        Input sites data
    bins : int, optional
        Number of bins
    n_parts : int
        Number of parts for error analysis

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    sites = jumps.sites

    maxlen = len(sites.trajectory) / n_parts
    binsize = maxlen / bins + 1
    data = []

    for jumps_part in jumps.split(n_parts=n_parts):
        data.append(
            np.histogram(jumps_part.as_dataframe()['start time'],
                         bins=bins,
                         range=(0., maxlen))[0])

    df = pd.DataFrame(data=data)
    columns = [binsize / 2 + binsize * col for col in range(bins)]

    mean = [df[col].mean() for col in df.columns]
    std = [df[col].std() for col in df.columns]

    df = pd.DataFrame(data=zip(columns, mean, std),
                      columns=['time', 'count', 'std'])
    df['specie'] = sites.floating_specie

    if n_parts > 1:
        fig = px.bar(df, x='time', y='count', color='specie', error_y='std')
    else:
        fig = px.bar(df, x='time', y='count', color='specie')

    fig.update_layout(bargap=0.2,
                      title='Jumps vs. time',
                      xaxis_title='Time (steps)',
                      yaxis_title='Number of jumps')

    return fig
