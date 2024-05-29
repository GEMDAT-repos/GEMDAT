from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:
    from gemdat import Jumps


def jumps_vs_time(*, jumps: Jumps, bins: int = 8, n_parts: int = 1) -> go.Figure:
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
    fig : plotly.graph_objects.Figure
        Output figure
    """
    maxlen = len(jumps.trajectory) / n_parts
    binsize = maxlen / bins + 1
    data = []

    for jumps_part in jumps.split(n_parts=n_parts):
        data.append(
            np.histogram(jumps_part.data['start time'], bins=bins, range=(0.0, maxlen))[0]
        )

    df = pd.DataFrame(data=data)
    columns = [binsize / 2 + binsize * col for col in range(bins)]

    mean = [df[col].mean() for col in df.columns]
    std = [df[col].std() for col in df.columns]

    df = pd.DataFrame(data=zip(columns, mean, std), columns=['time', 'count', 'std'])

    if n_parts > 1:
        fig = px.bar(df, x='time', y='count', error_y='std')
    else:
        fig = px.bar(df, x='time', y='count')

    fig.update_layout(
        bargap=0.2,
        title='Jumps vs. time',
        xaxis_title='Time (steps)',
        yaxis_title='Number of jumps',
    )

    return fig
