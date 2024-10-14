from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from .._shared import _get_vibrational_amplitudes_hist

if TYPE_CHECKING:
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
    metrics = trajectory.metrics()

    trajectories = trajectory.split(n_parts)

    amplitudes, counts, std = _get_vibrational_amplitudes_hist(
        trajectories=trajectories, bins=bins
    )

    min_amp = amplitudes.min()
    max_amp = amplitudes.max()

    # offset to middle of bar for plotly
    offset = (max_amp - min_amp) / (bins * 2)
    amplitudes += offset

    df = pd.DataFrame(data=zip(amplitudes, counts, std), columns=['amplitude', 'count', 'std'])

    if n_parts == 1:
        fig = px.bar(df, x='amplitude', y='count')
    else:
        fig = px.bar(df, x='amplitude', y='count', error_y='std')

    x = np.linspace(min_amp, max_amp, 100) + offset
    y_gauss = stats.norm.pdf(x, 0, metrics.vibration_amplitude())
    fig.add_trace(go.Scatter(x=x, y=y_gauss, name='Fitted Gaussian'))

    fig.update_layout(
        title='Histogram of vibrational amplitudes with fitted Gaussian',
        xaxis_title='Amplitude (Å)',
        yaxis_title='Occurrence (a.u.)',
    )

    return fig
