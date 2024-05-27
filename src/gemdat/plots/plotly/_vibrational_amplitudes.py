from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from gemdat.simulation_metrics import SimulationMetrics
from gemdat.trajectory import Trajectory


def vibrational_amplitudes(
    *, trajectory: Trajectory, bins: int = 50, n_parts: int = 1
) -> go.Figure:
    """Plot histogram of vibrational amplitudes with fitted Gaussian.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory, i.e. for the diffusing atom
    n_parts : int
        Number of parts for error analysis

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """

    trajectories = trajectory.split(n_parts)
    single_metrics = SimulationMetrics(trajectory)
    metrics = [
        SimulationMetrics(trajectory).amplitudes() for trajectory in trajectories
    ]

    max_amp = max(max(metric) for metric in metrics)
    min_amp = min(min(metric) for metric in metrics)

    max_amp = max(abs(min_amp), max_amp)
    min_amp = -max_amp

    data = []

    for metric in metrics:
        data.append(
            np.histogram(metric, bins=bins, range=(min_amp, max_amp), density=True)[0]
        )

    df = pd.DataFrame(data=data)

    # offset to middle of bar
    offset = (max_amp - min_amp) / (bins * 2)

    columns = np.linspace(min_amp + offset, max_amp + offset, bins, endpoint=False)

    mean = [df[col].mean() for col in df.columns]
    std = [df[col].std() for col in df.columns]

    df = pd.DataFrame(
        data=zip(columns, mean, std), columns=['amplitude', 'count', 'std']
    )

    if n_parts == 1:
        fig = px.bar(df, x='amplitude', y='count')
    else:
        fig = px.bar(df, x='amplitude', y='count', error_y='std')

    x = np.linspace(min_amp, max_amp, 100)
    y_gauss = stats.norm.pdf(x, 0, single_metrics.vibration_amplitude())
    fig.add_trace(go.Scatter(x=x, y=y_gauss, name='Fitted Gaussian'))

    fig.update_layout(
        title='Histogram of vibrational amplitudes with fitted Gaussian',
        xaxis_title='Amplitude (Å)',
        yaxis_title='Occurrence (a.u.)',
    )

    return fig
