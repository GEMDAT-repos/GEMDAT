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
    jumps : Jumps
        Input jumps data
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

    if n_parts == 1:
        fig = px.bar(df, x='Displacement', y='mean', barmode='stack')
    else:
        fig = px.bar(df,
                     x='Displacement',
                     y='mean',
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
    jumps : Jumps
        Input jumps data
    bins : int, optional
        Number of bins
    n_parts : int
        Number of parts for error analysis

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    maxlen = len(jumps.trajectory) / n_parts
    binsize = maxlen / bins + 1
    data = []

    for jumps_part in jumps.split(n_parts=n_parts):
        data.append(
            np.histogram(jumps_part.data['start time'],
                         bins=bins,
                         range=(0., maxlen))[0])

    df = pd.DataFrame(data=data)
    columns = [binsize / 2 + binsize * col for col in range(bins)]

    mean = [df[col].mean() for col in df.columns]
    std = [df[col].std() for col in df.columns]

    df = pd.DataFrame(data=zip(columns, mean, std),
                      columns=['time', 'count', 'std'])

    if n_parts > 1:
        fig = px.bar(df, x='time', y='count', error_y='std')
    else:
        fig = px.bar(df, x='time', y='count')

    fig.update_layout(bargap=0.2,
                      title='Jumps vs. time',
                      xaxis_title='Time (steps)',
                      yaxis_title='Number of jumps')

    return fig


def collective_jumps(*, jumps: Jumps) -> go.Figure:
    """Plot collective jumps per jump-type combination.

    Parameters
    ----------
    jumps : Jumps
        Input data

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """

    matrix = jumps.collective().site_pair_count_matrix()
    labels = jumps.collective().site_pair_count_matrix_labels()

    fig = px.imshow(matrix)

    ticks = [_ for _ in range(len(labels))]

    fig.update_layout(xaxis=dict(tickmode='array',
                                 tickvals=ticks,
                                 ticktext=labels),
                      yaxis=dict(tickmode='array',
                                 tickvals=ticks,
                                 ticktext=labels),
                      title='Cooperative jumps per jump-type combination')

    return fig


def jumps_3d(*, jumps: Jumps) -> go.Figure:
    """Plot jumps in 3D.

    Parameters
    ----------
    jumps : Jumps
        Input data

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    from ._density import plot_structure
    trajectory = jumps.trajectory
    sites = jumps.sites

    fig = go.Figure()
    plot_structure(sites, fig=fig)

    coords = sites.frac_coords
    lattice = trajectory.get_lattice()

    for i, j in zip(*np.triu_indices(len(coords), k=1)):
        count = jumps.matrix()[i, j] + jumps.matrix()[j, i]
        if count == 0:
            continue

        coord_i = coords[i]
        coord_j = coords[j]

        lw = 1 + np.log(count)

        length, image = lattice.get_distance_and_image(coord_i, coord_j)

        # NOTE: might need to plot `line = [coord_i - image, coord_j]` as well
        #if np.any(image != 0):
        #    continue
        lines = [[coord_i, coord_j + image], [coord_i - image, coord_j]]
        for line in lines:
            line = lattice.get_cartesian_coords(line)
            line = [_ for _ in zip(*line)]  # transpose, but pythonic

            fig.add_trace(
                go.Scatter3d(
                    x=line[0],
                    y=line[1],
                    z=line[2],
                    mode='lines',
                    showlegend=False,
                    line_dash='dashdot' if any(image) != 0 else 'solid',
                    line_width=lw * 3,
                    line_color='black',
                ))

    zoom = 0.1

    fig.update_layout(title='Jumps between sites',
                      scene={
                          'aspectmode': 'manual',
                          'aspectratio': {
                              'x': lattice.a * zoom,
                              'y': lattice.b * zoom,
                              'z': lattice.c * zoom,
                          },
                          'xaxis_title': 'X (Ångstrom)',
                          'yaxis_title': 'Y (Ångstrom)',
                          'zaxis_title': 'Z (Ångstrom)'
                      },
                      showlegend=True,
                      margin={
                          'l': 0,
                          'r': 0,
                          'b': 0,
                          't': 0
                      },
                      scene_camera={
                          'projection': {
                              'type': 'orthographic'
                          },
                          'eye': {
                              'x': -lattice.a * 0.05,
                              'y': -lattice.b * 0.2,
                              'z': lattice.c * 0.15,
                          }
                      })

    return fig
