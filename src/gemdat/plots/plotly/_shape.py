from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Collection

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from pymatgem.core import PeriodicSite

    from gemdat.shape import ShapeData


def shape(
    shape: ShapeData,
    nbins: int = 50,
    sites: Collection[PeriodicSite] | None = None,
) -> go.Figure:
    """Plot site cluster shapes.

    Parameters
    ----------
    shape : ShapeData
        Shape data to plot
    nbins : int
        Number of bins
    sites : Collection[PeriodicSite] | None
        Plot these sites on the shape density

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """
    x_labels = ('X / Å', 'Y / Å', 'Z / Å')
    y_labels = ('Y / Å', 'Z / Å', 'X / Å')

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=x_labels,
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    distances = shape.distances()
    distances_sq = distances**2

    msd = np.mean(distances_sq)
    std = np.std(distances_sq)
    # Latex can not be rendered in 3D plotly plots: https://github.com/plotly/plotly.js/issues/608
    title = f'{shape.name}: MSD = {msd:.3f} Å^2, std = {std:.3f}'

    mean_dist = np.mean(distances)

    _tmp_dict = defaultdict(list)
    vector_dict = {}
    if sites:
        for site in sites:
            _tmp_dict[site.label].append(site.coords)
        for key, values in _tmp_dict.items():
            vector_dict[key] = np.array(values) - shape.origin

    coords = shape.coords

    for col, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        x_coords = coords[:, i]
        y_coords = coords[:, j]
        top_row = {'row': 1, 'col': col + 1}
        bot_row = {'row': 2, 'col': col + 1}
        dot_marker = {'color': 'red', 'line': {'width': 0.5, 'color': 'White'}}
        dashed_line = {'color': 'red', 'dash': 'dash'}

        # Map of the coordinates
        fig.add_trace(
            go.Histogram2dContour(
                x=x_coords,
                y=y_coords,
                name=y_labels[col],
                showscale=False,
            ),
            **top_row,
        )

        # Highlight the (0,0,0) point
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode='markers',
                marker=dot_marker,
                name='Sites ' + y_labels[col],
                showlegend=False,
            ),
            **top_row,
        )
        # Area around the zero
        fig.add_shape(
            type='circle',
            xref='x',
            yref='y',
            x0=-mean_dist,
            y0=-mean_dist,
            x1=mean_dist,
            y1=mean_dist,
            line=dashed_line,
            showlegend=False,
            **top_row,
        )

        for label, vects in vector_dict.items():
            x_vs = vects[:, i]
            y_vs = vects[:, j]

            # Add shape points
            fig.add_trace(
                go.Scatter(
                    x=x_vs,
                    y=y_vs,
                    mode='markers+text',
                    text=[label] * len(x_vs),
                    textposition='top center',
                    textfont={'color': 'red'},
                    marker={'color': 'red'},
                    name=label,
                    showlegend=False,
                ),
                **top_row,
            )
            # Area around the shape points
            fig.add_shape(
                type='circle',
                xref='x',
                yref='y',
                x0=x_vs - mean_dist,
                y0=y_vs - mean_dist,
                x1=x_vs + mean_dist,
                y1=x_vs + mean_dist,
                line=dashed_line,
                showlegend=False,
                **top_row,
            )

        # Density plot along the coordinate
        fig.add_trace(
            go.Histogram(
                x=x_coords,
                name=x_labels[col],
                nbinsx=nbins,
                histnorm='probability density',
            ),
            **bot_row,
        )

    fig.update_layout(height=600, width=900, title_text=title, title_x=0.5)

    return fig
