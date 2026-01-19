from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from gemdat.plots._shared import hex2rgba


if TYPE_CHECKING:
    from gemdat.metrics import ArrheniusFit

def arrhenius(*, fit: ArrheniusFit, show_std: bool = True) -> go.Figure:
    """Plot Arrhenius fit.

    Parameters
    ----------
    fit
        ArrheniusFit instance.
    show_std
        If True, show error bars (from diffusivities_std) and a ±1σ fit band
        (from cov).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Output figure
    """

    T = fit.temperatures
    x = 1000.0 / T
    y = np.log(fit.diffusivities)

    error_y = None
    if show_std and getattr(fit, 'diffusivities_std', None) is not None:
        sigma_ln = fit.diffusivities_std / fit.diffusivities
        error_y = dict(type='data', array=sigma_ln, visible=True)

    fig = go.Figure()
    color_hex = fig.layout['template']['layout']['colorway'][0]
    color_rgba = hex2rgba(color_hex, opacity=0.3)
    
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='data', error_y=error_y, line_color=color_hex))

    # Fit line
    t_line = np.linspace(float(T.min()), float(T.max()), 200)
    x_line = 1000.0 / t_line
    ln_line = fit.intercept + fit.slope * (1.0 / t_line)
    fig.add_trace(go.Scatter(x=x_line, y=ln_line, mode='lines', name='fit', line_color=color_hex))

    # ±1σ band (in ln-space)
    if show_std and getattr(fit, 'cov', None) is not None:
        v = np.column_stack([1.0 / t_line, np.ones_like(t_line)])
        var = np.einsum('ij,jk,ik->i', v, fit.cov, v)
        std = np.sqrt(np.maximum(var, 0.0))
        upper = ln_line + std
        lower = ln_line - std

        fig.add_trace(go.Scatter(x=x_line, y=upper, mode='lines', line=dict(width=0), showlegend=False, fillcolor=color_rgba))
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=lower,
                mode='lines',
                fill='tonexty',
                line=dict(width=0),
                name='±1σ',
                opacity=0.2,
                fillcolor=color_rgba,
            )
        )

    fig.update_layout(xaxis_title='1000/T (K⁻¹)', yaxis_title='ln(D)')
    return fig

