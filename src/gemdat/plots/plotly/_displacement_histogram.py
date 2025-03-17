from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:
    from gemdat.trajectory import Trajectory


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
        trajectory.species, trajectory.distances_from_base_position()[:, -1]
    ):
        data.append((specie, round(distance)))

    df = pd.DataFrame(columns=['Element', 'Displacement'], data=data)
    df = (
        df.groupby(['Displacement', 'Element'])
        .size()
        .reset_index()
        .rename(columns={0: 'count'})
    )
    return df


def displacement_histogram(trajectory: Trajectory, n_parts: int = 1) -> go.Figure:
    """Plot histogram of total displacement at final timestep.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom
    n_parts : int
        Plot error bars by dividing data into n parts

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    if n_parts == 1:
        df = _trajectory_to_dataframe(trajectory)

        fig = px.bar(df, x='Displacement', y='count', color='Element', barmode='stack')

        fig.update_layout(
            title='Displacement per element',
            xaxis_title='Displacement (Å)',
            yaxis_title='Nr. of atoms',
        )
    else:
        interval = np.linspace(0, len(trajectory) - 1, n_parts + 1)
        dfs = [_trajectory_to_dataframe(part) for part in trajectory.split(n_parts)]

        all_df = pd.concat(dfs)

        # Get the mean and standard deviation
        grouped = all_df.groupby(['Displacement', 'Element'])
        mean = grouped.mean().reset_index().rename(columns={'count': 'mean'})
        std = grouped.std().reset_index().rename(columns={'count': 'std'})
        df = mean.merge(std, how='inner')

        fig = px.bar(
            df,
            x='Displacement',
            y='mean',
            color='Element',
            error_y='std',
            barmode='group',
        )

        fig.update_layout(
            title=(
                f'Displacement per element after {int(interval[1] - interval[0])} timesteps'
            ),
            xaxis_title='Displacement (Å)',
            yaxis_title='Nr. of atoms',
        )

    return fig
