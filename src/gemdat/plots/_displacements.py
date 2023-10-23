from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gemdat.trajectory import Trajectory


def displacement_per_site(*, trajectory: Trajectory) -> plt.Figure:
    """Plot displacement per site.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()

    for distances in trajectory.distances_from_base_position():
        ax.plot(distances, lw=0.3)

    ax.set(title='Displacement per site',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    return fig


def displacement_per_element(*, trajectory: Trajectory) -> plt.Figure:
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
    from collections import defaultdict

    grouped = defaultdict(list)

    species = trajectory.species

    for sp, distances in zip(species,
                             trajectory.distances_from_base_position()):
        grouped[sp.symbol].append(distances)

    fig, ax = plt.subplots()

    for symbol, distances in grouped.items():
        mean_disp = np.mean(distances, axis=0)
        ax.plot(mean_disp, lw=0.3, label=symbol)

    ax.legend()
    ax.set(title='Displacement per element',
           xlabel='Time step',
           ylabel='Displacement (Angstrom)')

    return fig


def displacement_per_element2(*, trajectory: Trajectory) -> go.Figure:
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


def displacement_histogram(trajectory: Trajectory) -> plt.Figure:
    """Plot histogram of total displacement at final timestep.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """
    fig, ax = plt.subplots()
    ax.hist(trajectory.distances_from_base_position()[:, -1])
    ax.set(title='Histogram of displacements',
           xlabel='Displacement (Angstrom)',
           ylabel='Nr. of atoms')

    return fig


def displacement_histogram2(trajectory: Trajectory) -> go.Figure:
    """Plot histogram of total displacement at final timestep.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom

    Returns
    -------
    fig : matplotlib.figure.Figure
        Output figure
    """

    defaultdict(list)

    species = trajectory.species

    data = []
    for specie, distance in zip(
            species,
            trajectory.distances_from_base_position()[:, -1]):
        data.append((specie, round(distance)))

    df = pd.DataFrame(columns=['Element', 'Displacement'], data=data)
    df = df.groupby(['Displacement', 'Element'
                     ]).size().reset_index().rename(columns={0: 'count'})

    fig = px.bar(df,
                 x='Displacement',
                 y='count',
                 color='Element',
                 barmode='stack')

    fig.update_layout(title='Displacement per element',
                      xaxis_title='Displacement (Angstrom)',
                      yaxis_title='Nr. of atoms')

    return fig
