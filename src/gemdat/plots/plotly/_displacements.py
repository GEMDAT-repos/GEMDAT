from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gemdat.trajectory import Trajectory


def displacement_per_atom(*, trajectory: Trajectory) -> go.Figure:
    """Plot displacement per atom.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    fig = go.Figure()

    distances = [dist for dist in trajectory.distances_from_base_position()]

    for i, distance in enumerate(distances):
        fig.add_trace(
            go.Scatter(y=distance,
                       name=i,
                       mode='lines',
                       line={'width': 1},
                       showlegend=False))

    fig.update_layout(title='Displacement per atom',
                      xaxis_title='Time step',
                      yaxis_title='Displacement (Angstrom)')

    return fig


def displacement_per_element(*, trajectory: Trajectory) -> go.Figure:
    """Plot displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    fig = go.Figure()

    grouped = defaultdict(list)

    species = trajectory.species

    for sp, distances in zip(species,
                             trajectory.distances_from_base_position()):
        grouped[sp.symbol].append(distances)

    for symbol, distances in grouped.items():
        mean_disp = np.mean(distances, axis=0)
        std_disp = np.std(distances, axis=0)
        fig.add_trace(
            go.Scatter(y=mean_disp,
                       name=symbol + ' + std',
                       mode='lines',
                       line={'width': 3},
                       legendgroup=symbol))
        fig.add_trace(
            go.Scatter(y=mean_disp + std_disp,
                       name=symbol + ' + std',
                       mode='lines',
                       line={'width': 0},
                       legendgroup=symbol,
                       showlegend=False))
        fig.add_trace(
            go.Scatter(y=mean_disp - std_disp,
                       name=symbol + ' + std',
                       mode='lines',
                       line={'width': 0},
                       legendgroup=symbol,
                       showlegend=False,
                       fill='tonexty'))

    fig.update_layout(title='Displacement per element',
                      xaxis_title='Time step',
                      yaxis_title='Displacement (Angstrom)')

    return fig


def msd_per_element(*, trajectory: Trajectory) -> go.Figure:
    """Plot mean squared displacement per element.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    fig = go.Figure()

    species = list(set(trajectory.species))

    for sp in species:
        traj = trajectory.filter(sp.symbol)

        fig.add_trace(
            go.Scatter(y=traj.mean_squared_displacement().mean(axis=0),
                       name=sp.symbol,
                       mode='lines',
                       line={'width': 3},
                       legendgroup=sp.symbol))

    fig.update_layout(title='Mean squared displacement per element',
                      xaxis_title='Time step',
                      yaxis_title='MSD (Angstrom<sup>2</sup>)')

    return fig


def _trajectory_to_dataframe(trajectory: Trajectory) -> pd.DataFrame:
    """_trajectory_to_dataframe.

    Parameters
    ----------
    trajectory : Trajectory
        trajectory

    Returns
    -------
    pd.DataFrame
    """
    data = []
    for specie, distance in zip(
            trajectory.species,
            trajectory.distances_from_base_position()[:, -1]):
        data.append((specie, round(distance)))

    df = pd.DataFrame(columns=['Element', 'Displacement'], data=data)
    df = df.groupby(['Displacement', 'Element'
                     ]).size().reset_index().rename(columns={0: 'count'})
    return df


def displacement_histogram(trajectory: Trajectory,
                           n_parts: int = 1) -> go.Figure:
    """Plot histogram of total displacement at final timestep.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom
    n_parts : int
        Plot error bars by dividing data into n parts

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    if n_parts == 1:
        df = _trajectory_to_dataframe(trajectory)

        fig = px.bar(df,
                     x='Displacement',
                     y='count',
                     color='Element',
                     barmode='stack')

        fig.update_layout(title='Displacement per element',
                          xaxis_title='Displacement (Angstrom)',
                          yaxis_title='Nr. of atoms')
    else:
        interval = np.linspace(0, len(trajectory) - 1, n_parts + 1)
        dfs = [
            _trajectory_to_dataframe(part)
            for part in trajectory.split(n_parts)
        ]

        all_df = pd.concat(dfs)

        # Get the mean and standard deviation
        grouped = all_df.groupby(['Displacement', 'Element'])
        mean = grouped.mean().reset_index().rename(columns={'count': 'mean'})
        std = grouped.std().reset_index().rename(columns={'count': 'std'})
        df = mean.merge(std, how='inner')

        fig = px.bar(df,
                     x='Displacement',
                     y='mean',
                     color='Element',
                     error_y='std',
                     barmode='group')

        fig.update_layout(
            title=
            f'Displacement per element after {int(interval[1]-interval[0])} timesteps',
            xaxis_title='Displacement (Angstrom)',
            yaxis_title='Nr. of atoms')

    return fig
